"""
highlight.py
============
Turns model explanations into something a therapist can read at a glance:

  * the original text with every influential word coloured
    (red = pushed towards depression, green = pushed away from it,
    stronger colour = stronger effect), combining both models;
  * recognised warning phrases underlined and labelled with a theme
    (loss of interest, hopelessness, low energy, ...), based on the
    curated phrase list in src/features/clinical_lexicon.py;
  * a plain-English "why" sentence and a key-words table.

Only standard library + markupsafe (ships with Flask).
"""
from __future__ import annotations

import html
import re
from collections import OrderedDict

# Warning phrases grouped by theme. Same phrases as
# src/features/clinical_lexicon.py (grouped there by comments), plus a few
# common variants. Themes loosely follow PHQ-9 symptom areas.
THEMES: "OrderedDict[str, list[str]]" = OrderedDict([
    ("Loss of interest", ["nothing feels worth", "don't enjoy", "lost interest", "nothing matters",
                          "no point in", "what's the point", "no motivation", "don't care anymore"]),
    ("Hopelessness", ["no hope", "never get better", "nothing will change", "give up", "giving up",
                      "no way out", "hopeless"]),
    ("Low self-worth", ["i'm worthless", "im worthless", "feel worthless", "such a burden",
                        "everyone would be better", "hate myself", "not good enough"]),
    ("Low energy", ["can't get out of bed", "cant get out of bed", "can't get up", "no energy",
                    "too tired to", "can't function", "can't do anything", "exhausted all the time"]),
    ("Sleep problems", ["can't sleep", "cant sleep", "sleep all day", "awake all night"]),
    ("Withdrawal from people", ["stopped talking to", "don't want to see anyone", "isolate myself",
                                "isolated myself", "cancelled on everyone", "no one to talk to"]),
    ("Numbness / emptiness", ["feel numb", "feel empty", "feel nothing", "empty inside"]),
    ("Thoughts of disappearing", ["want to disappear", "wish i could disappear", "want to sleep forever",
                                  "don't want to exist", "dont want to exist", "want it to end"]),
])
# Themes that should trigger an "urgent" note in the report.
URGENT_THEMES = {"Thoughts of disappearing"}

# Fragments LIME produces when it splits on apostrophes ("can't" -> "can", "t").
# They are never coloured on their own.
FRAGMENTS = {"t", "s", "m", "re", "ve", "ll", "d"}
# Very common words left out of the plain-English sentence (still shown in
# the table, so the model's behaviour stays visible to the therapist).
FUNCTION_WORDS = {"i", "me", "my", "a", "an", "the", "and", "or", "to", "of", "in", "on", "it", "is", "am",
                  "are", "was", "be", "but", "so", "that", "this", "for", "with", "at", "as", "just"}

_APOS = str.maketrans({"’": "'", "‘": "'", "`": "'"})
# Same word split LIME uses by default (split on non-word characters).
_SPLIT = re.compile(r"(\W+)", re.UNICODE)


def _norm(text: str) -> str:
    """Lower-case and straighten apostrophes, keeping the same length."""
    return text.translate(_APOS).lower()


def find_themes(text: str) -> list[dict]:
    """Return non-overlapping phrase matches: {theme, phrase, start, end}."""
    low = _norm(text)
    found = []
    for theme, phrases in THEMES.items():
        for ph in phrases:
            variants = {ph, ph.replace("'", "")}
            for v in variants:
                for m in re.finditer(r"(?<!\w)" + re.escape(v) + r"(?!\w)", low):
                    found.append({"theme": theme, "phrase": text[m.start():m.end()],
                                  "start": m.start(), "end": m.end()})
    found.sort(key=lambda f: (f["start"], -(f["end"] - f["start"])))
    out, last_end = [], -1
    for f in found:
        if f["start"] >= last_end:
            out.append(f)
            last_end = f["end"]
    return out


def towards_depression(model_result: dict) -> dict:
    """LIME weights are relative to the model's predicted class. Convert them
    so that positive always means 'pushed towards depression'."""
    sign = 1.0 if model_result.get("prediction") == "depression" else -1.0
    out = {}
    for word, val in model_result.get("lime_words") or []:
        out[str(word)] = out.get(str(word), 0.0) + sign * float(val)
    return out


def combine(model_breakdown: list[dict]) -> tuple[dict, list[dict]]:
    """Average 'towards depression' weights across the models.
    Returns (combined {word: weight}, per-model list of dicts)."""
    per_model = [towards_depression(m) for m in model_breakdown]
    words = set().union(*per_model) if per_model else set()
    n = max(1, len(per_model))
    combined = {w: sum(pm.get(w, 0.0) for pm in per_model) / n for w in words}
    return combined, per_model


def highlight_html(text: str, weights: dict, themes: list[dict] | None = None) -> str:
    """Return safe HTML of ``text`` with influential words coloured and
    theme phrases underlined. ``weights``: {word: towards-depression weight}."""
    themes = themes if themes is not None else find_themes(text)
    max_w = max([abs(v) for v in weights.values()] + [1e-9])
    # character -> theme label at phrase start / inside phrase
    starts = {t["start"]: t for t in themes}
    inside = [None] * len(text)
    for t in themes:
        for i in range(t["start"], t["end"]):
            inside[i] = t
    parts, pos, open_theme = [], 0, None
    for tok in _SPLIT.split(text):
        if not tok:
            continue
        start = pos
        pos += len(tok)
        t_here = inside[start] if start < len(inside) else None
        if open_theme is not None and t_here is not open_theme:
            parts.append(f'<sup class="hl-tag">{html.escape(open_theme["theme"])}</sup></span>')
            open_theme = None
        if t_here is not None and open_theme is None and start in starts:
            open_theme = t_here
            parts.append('<span class="hl-phrase">')
        esc = html.escape(tok)
        w = weights.get(tok)
        if w is None or _SPLIT.fullmatch(tok) or tok.lower() in FRAGMENTS or abs(w) < 0.02 * max_w:
            parts.append(esc)
            continue
        strength = min(1.0, abs(w) / max_w)
        alpha = 0.18 + 0.62 * strength
        cls = "hl-dep" if w > 0 else "hl-ok"
        rgb = "192,57,43" if w > 0 else "46,125,82"
        title = ("pushed towards depression" if w > 0 else "pushed away from depression") + f" (weight {w:+.3f})"
        parts.append(f'<mark class="{cls}" style="background:rgba({rgb},{alpha:.2f})" '
                     f'title="{title}">{esc}</mark>')
    if open_theme is not None:
        parts.append(f'<sup class="hl-tag">{html.escape(open_theme["theme"])}</sup></span>')
    return "".join(parts)


def why_sentence(prediction_word: str, combined: dict, themes: list[dict]) -> str:
    """One plain-English sentence explaining the main reasons."""
    keep = {w: v for w, v in combined.items() if w.lower() not in FUNCTION_WORDS | FRAGMENTS}
    up = [w for w, v in sorted(keep.items(), key=lambda kv: -kv[1]) if v > 0][:4]
    down = [w for w, v in sorted(keep.items(), key=lambda kv: kv[1]) if v < 0][:3]
    bits = []
    if up:
        bits.append("the words " + ", ".join(f"“{w}”" for w in up)
                    + " pushed the AI towards depression")
    if down:
        bits.append("“" + "”, “".join(down) + "” pointed the other way")
    s = "Mainly, " + "; ".join(bits) + "." if bits else "No single word stood out strongly."
    th = list(OrderedDict.fromkeys(t["theme"].lower() for t in themes))
    if th:
        s += " The text also contains phrases linked to " + _join(th) + "."
    return s


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def explain(text: str, model_breakdown: list[dict], prediction_word: str = "") -> dict:
    """Everything the report template needs for one text."""
    combined, per_model = combine(model_breakdown)
    themes = find_themes(text)
    rows = []
    for w in [k for k in sorted(combined, key=lambda k: -abs(combined[k])) if k.lower() not in FRAGMENTS][:8]:
        rows.append({"word": w, "combined": combined[w],
                     "per_model": [pm.get(w) for pm in per_model]})
    max_w = max([abs(r["combined"]) for r in rows] + [1e-9])
    for r in rows:
        r["bar"] = round(100 * abs(r["combined"]) / max_w)
    return {
        "html": highlight_html(text, combined, themes),
        "why": why_sentence(prediction_word, combined, themes),
        "themes": themes,
        "theme_names": list(OrderedDict.fromkeys(t["theme"] for t in themes)),
        "urgent": any(t["theme"] in URGENT_THEMES for t in themes),
        "rows": rows,
        "model_names": [m.get("model", f"Model {i + 1}") for i, m in enumerate(model_breakdown)],
    }
