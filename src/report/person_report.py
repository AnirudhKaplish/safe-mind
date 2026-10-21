"""
person_report.py
================
One report from MANY posts by the same person (e.g. their Reddit history).

Input (consent-based, the person shares their own data):
  * a Reddit data export file - posts.csv / comments.csv from
    reddit.com/settings/data-request (columns include date, subreddit,
    title, body), or
  * a JSON list of {"text": ..., "date": ...}, or
  * plain text with posts separated by a blank line or a line of "---".

Pipeline:
  1. parse posts, drop empty / deleted ones
  2. run the personal-post filter on each (too short, question, about
     someone else, ...) - skipped posts are listed, not scored
  3. score every remaining post with both models (batched, fast)
  4. explain only the most concerning posts word-by-word (LIME is slow)
  5. aggregate: overall verdict, timeline, recurring themes, recurring
     words (exact per-word contributions from the linear baseline model)

The overall verdict uses a simple, stated rule (see ``overall_verdict``)
so a therapist can see exactly why the person-level result was given.
"""
from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
from typing import Callable, Optional

from src.features.input_type_filter import classify_input_type
from src.xai import highlight

MAX_POSTS = 200            # analyse at most this many (most recent first)
EXPLAIN_TOP = 3            # word-level explanations for this many posts
DELETED = {"[deleted]", "[removed]", ""}


# ---------------------------------------------------------------------------
# 1. Parsing
# ---------------------------------------------------------------------------
def _parse_date(s) -> Optional[datetime]:
    if s is None:
        return None
    s = str(s).strip().replace(" UTC", "").replace("Z", "")
    if re.fullmatch(r"\d{9,11}(\.\d+)?", s):           # unix timestamp
        try:
            return datetime.utcfromtimestamp(float(s))
        except (OverflowError, ValueError):
            return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def _post(text, date=None, source="", subreddit=""):
    text = (text or "").strip()
    return {"text": text, "date": _parse_date(date), "source": source, "subreddit": subreddit or ""}


def parse_csv(raw: str, source: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(raw)))
    posts = []
    for r in rows:
        r = {(k or "").strip().lower(): (v or "") for k, v in r.items()}
        title = r.get("title", "").strip()
        body = (r.get("body") or r.get("text") or r.get("selftext") or r.get("content") or "").strip()
        if body in DELETED:
            body = ""
        text = (title + ". " + body).strip(". ").strip() if title and body else (title or body)
        posts.append(_post(text, r.get("date") or r.get("created_utc") or r.get("created") or r.get("timestamp"),
                           source, r.get("subreddit", "")))
    return posts


def parse_json(raw: str, source: str) -> list[dict]:
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("posts") or data.get("data") or []
    posts = []
    for item in data:
        if isinstance(item, str):
            posts.append(_post(item, source=source))
        elif isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            body = str(item.get("text") or item.get("body") or item.get("selftext") or "").strip()
            text = (title + ". " + body).strip(". ").strip() if title and body else (title or body)
            posts.append(_post(text, item.get("date") or item.get("created_utc"), source,
                               item.get("subreddit", "")))
    return posts


def parse_text(raw: str, source: str = "pasted") -> list[dict]:
    chunks = re.split(r"\n\s*(?:-{3,})?\s*\n", raw.replace("\r\n", "\n"))
    posts = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        # optional "2024-05-01: text" or "[2024-05-01] text" date prefix
        m = re.match(r"^\[?(\d{4}-\d{2}-\d{2})\]?\s*[:\-]?\s+(.*)$", c, re.S)
        posts.append(_post(m.group(2), m.group(1), source) if m else _post(c, source=source))
    return posts


def parse_any(raw: str, filename: str = "") -> list[dict]:
    name = (filename or "").lower()
    head = raw.lstrip()[:1]
    if name.endswith(".json") or head in ("[", "{"):
        return parse_json(raw, filename or "json")
    first_line = raw.splitlines()[0].lower() if raw.strip() else ""
    if name.endswith(".csv") or ("," in first_line and any(k in first_line for k in ("body", "title", "text"))):
        return parse_csv(raw, filename or "csv")
    return parse_text(raw, filename or "pasted")


def clean_posts(posts: list[dict]) -> list[dict]:
    posts = [p for p in posts if p["text"] and p["text"] not in DELETED]
    seen, out = set(), []
    for p in posts:                      # drop exact duplicates
        if p["text"] in seen:
            continue
        seen.add(p["text"])
        out.append(p)
    dated = [p for p in out if p["date"]]
    if len(dated) == len(out) and out:
        out.sort(key=lambda p: p["date"])
    if len(out) > MAX_POSTS:
        out = out[-MAX_POSTS:]
    for i, p in enumerate(out):
        p["n"] = i + 1
    return out


# ---------------------------------------------------------------------------
# 2-5. Analysis
# ---------------------------------------------------------------------------
def overall_verdict(n_analysed, n_flagged, n_unsure, urgent) -> dict:
    """Stated, simple rule for the person-level result."""
    if n_analysed == 0:
        return {"level": "none", "title": "Nothing could be checked",
                "rule": "None of the posts were long enough or personal enough to screen."}
    share = n_flagged / n_analysed
    if urgent or (n_flagged >= 2 and share >= 0.25):
        return {"level": "red", "title": "Repeated signs of depression across posts",
                "rule": "Shown when at least 2 posts, and at least 25% of checked posts, were flagged by "
                        "both models - or when any post mentions wanting to disappear."}
    if n_flagged >= 1 or n_unsure >= 2:
        return {"level": "amber", "title": "Some posts show possible signs",
                "rule": "Shown when at least 1 post was flagged by both models, or the models disagreed on "
                        "2 or more posts."}
    return {"level": "green", "title": "No clear signs of depression across posts",
            "rule": "No post was flagged by both models, and the models disagreed on fewer than 2 posts."}


def _trend(points: list[float]) -> Optional[str]:
    if len(points) < 6:
        return None
    k = max(2, len(points) // 3)
    first, last = sum(points[:k]) / k, sum(points[-k:]) / k
    if last - first > 0.15:
        return "rising"
    if first - last > 0.15:
        return "falling"
    return "steady"


def analyse(posts: list[dict], *, baseline_dep_probs: Callable[[list[str]], list[float]],
            bert_dep_probs: Optional[Callable[[list[str]], list[float]]],
            explain_post: Callable[[str], list[dict]],
            baseline_word_contribs: Optional[Callable[[list[str]], list[dict]]] = None,
            thresholds=(0.40, 0.50), explain_top: int = EXPLAIN_TOP) -> dict:
    """Score and summarise a person's posts. Model access is injected so this
    module stays independent of the dashboard."""
    posts = clean_posts(posts)
    skipped, todo = [], []
    for p in posts:
        kind = classify_input_type(p["text"])
        if kind["category"] != "personal_disclosure":
            p["skip_reason"] = kind["category"].replace("_", " ")
            skipped.append(p)
        else:
            todo.append(p)

    texts = [p["text"] for p in todo]
    b_probs = baseline_dep_probs(texts) if texts else []
    t_probs = bert_dep_probs(texts) if (texts and bert_dep_probs) else b_probs
    for p, b, t in zip(todo, b_probs, t_probs):
        b_dep, t_dep = b >= thresholds[0], t >= thresholds[1]
        p["base_prob"], p["bert_prob"] = float(b), float(t)
        p["avg_prob"] = (float(b) + float(t)) / 2
        p["status"] = "flagged" if (b_dep and t_dep) else "clear" if (not b_dep and not t_dep) else "unsure"
        p["themes"] = highlight.find_themes(p["text"])
        p["theme_names"] = list(OrderedDict.fromkeys(t["theme"] for t in p["themes"]))
        p["urgent"] = any(t["theme"] in highlight.URGENT_THEMES for t in p["themes"])

    # word-level explanations for the most concerning posts
    order = {"flagged": 0, "unsure": 1, "clear": 2}
    concerning = sorted([p for p in todo if p["status"] != "clear" or p["themes"]],
                        key=lambda p: (not p["urgent"], order[p["status"]], -p["avg_prob"]))
    attention = concerning[:explain_top]
    for p in attention:
        breakdown = explain_post(p["text"])
        p["explanation"] = highlight.explain(p["text"], breakdown)
        p["breakdown"] = breakdown

    n_flag = sum(p["status"] == "flagged" for p in todo)
    n_unsure = sum(p["status"] == "unsure" for p in todo)
    urgent = any(p["urgent"] for p in todo)
    verdict = overall_verdict(len(todo), n_flag, n_unsure, urgent)

    # recurring themes
    theme_posts = defaultdict(list)
    theme_phr = defaultdict(Counter)
    for p in todo:
        for t in p["themes"]:
            theme_phr[t["theme"]][t["phrase"].lower()] += 1
        for name in p["theme_names"]:
            theme_posts[name].append(p)
    themes = []
    for name in highlight.THEMES:
        if name in theme_posts:
            ps = theme_posts[name]
            dates = [q["date"] for q in ps if q["date"]]
            themes.append({"theme": name, "posts": len(ps), "urgent": name in highlight.URGENT_THEMES,
                           "phrases": [ph for ph, _ in theme_phr[name].most_common(4)],
                           "first": min(dates) if dates else None, "last": max(dates) if dates else None,
                           "post_numbers": [q["n"] for q in ps]})
    themes.sort(key=lambda t: (not t["urgent"], -t["posts"]))

    # recurring words: exact per-word contributions of the linear model, all posts
    words = []
    if baseline_word_contribs and texts:
        agg, cnt = defaultdict(float), Counter()
        for contrib in baseline_word_contribs(texts):
            for w, v in contrib.items():
                agg[w] += v
                cnt[w] += 1
        top = sorted([w for w in agg if agg[w] > 0], key=lambda w: -agg[w])[:12]
        mx = max([agg[w] for w in top] + [1e-9])
        words = [{"word": w, "posts": cnt[w], "bar": round(100 * agg[w] / mx)} for w in top]

    timeline = [{"n": p["n"], "prob": p["avg_prob"], "status": p["status"],
                 "date": p["date"].strftime("%d %b %Y") if p["date"] else ""} for p in todo]
    dated = all(p["date"] for p in todo) and todo
    return {
        "posts": posts, "analysed": todo, "skipped": skipped, "attention": attention,
        "counts": {"total": len(posts), "analysed": len(todo), "flagged": n_flag, "unsure": n_unsure,
                   "clear": len(todo) - n_flag - n_unsure, "skipped": len(skipped)},
        "verdict": verdict, "urgent": urgent, "themes": themes, "words": words,
        "timeline": timeline, "trend": _trend([p["avg_prob"] for p in todo]) if dated else None,
        "period": (todo[0]["date"].strftime("%d %b %Y"), todo[-1]["date"].strftime("%d %b %Y")) if dated else None,
        "subreddits": Counter(p["subreddit"] for p in todo if p["subreddit"]).most_common(5),
    }


def robot_report(summary: dict) -> dict:
    """Map the person-level verdict onto the single-report shape the robot
    understands (no text, IDs or words are included)."""
    lvl = summary["verdict"]["level"]
    c = summary["counts"]
    if lvl == "none":
        return {"status": "NOT_APPLICABLE"}
    if lvl == "amber":
        return {"status": "DISAGREEMENT"}
    share = c["flagged"] / c["analysed"] if c["analysed"] else 0.0
    if lvl == "red":
        return {"status": "AGREEMENT", "risk_class": "depression", "confidence": round(max(share, 0.5), 4)}
    return {"status": "AGREEMENT", "risk_class": "not_depression", "confidence": round(1 - share, 4)}


def timeline_svg(timeline: list[dict], width: int = 640, height: int = 170) -> str:
    """Small inline SVG: one dot per post (left = oldest), height = how
    strongly the two models leaned towards depression."""
    if not timeline:
        return ""
    pad_l, pad_r, pad_t, pad_b = 34, 10, 10, 26
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(timeline)
    xs = [pad_l + (w * i / (n - 1) if n > 1 else w / 2) for i in range(n)]
    ys = [pad_t + h * (1 - t["prob"]) for t in timeline]
    col = {"flagged": "#C0392B", "unsure": "#B08328", "clear": "#2E7D52"}
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Timeline of posts">']
    for v, lab in ((1.0, "100%"), (0.5, "50%"), (0.0, "0%")):
        y = pad_t + h * (1 - v)
        dash = ' stroke-dasharray="4 3"' if v == 0.5 else ""
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#DEDBD2"{dash}/>')
        out.append(f'<text x="{pad_l - 6}" y="{y + 4:.1f}" font-size="10" text-anchor="end" fill="#6B7672">{lab}</text>')
    if n > 1:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        out.append(f'<polyline points="{pts}" fill="none" stroke="#9AA39F" stroke-width="1.5"/>')
    for x, y, t in zip(xs, ys, timeline):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col[t["status"]]}">'
                   f'<title>Post {t["n"]} {t["date"]}: {t["prob"] * 100:.0f}%</title></circle>')
    first = timeline[0]["date"] or "first post"
    last = timeline[-1]["date"] or "latest post"
    out.append(f'<text x="{pad_l}" y="{height - 6}" font-size="10" fill="#6B7672">{first}</text>')
    out.append(f'<text x="{width - pad_r}" y="{height - 6}" font-size="10" text-anchor="end" fill="#6B7672">{last}</text>')
    out.append("</svg>")
    return "".join(out)
