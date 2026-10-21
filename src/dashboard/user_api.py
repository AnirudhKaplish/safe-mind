"""
user_api.py
===========
Connects the USER web app to the other two parts of SAFE-MIND:

  1. the screening engine  (pre-filter, baseline model, BERT, ensemble rule)
  2. the consent contract  (real transactions through blockchain/scripts/consent_server.js)
  3. the clinician dashboard + robot (the report can be shared to Meridian, and the
     robot state is updated live while the user's posts are being checked)

Wire it into src/dashboard/app.py with TWO lines placed just above `if __name__ == "__main__":`

    from src.dashboard.user_api import register
    register(app, baseline_dep_probs, bert_dep_probs if BERT_AVAILABLE else None, build_report, REPORTS,
             robot_bus, processing_command, from_report, error_command,
             (BASELINE_DEPRESSION_THRESHOLD, BERT_DEPRESSION_THRESHOLD))

Then open  http://127.0.0.1:5000/user
"""
import json
import os
import re
import sys
import urllib.request
from datetime import datetime

from flask import Blueprint, Response, jsonify, request, stream_with_context

CHAIN_URL = os.environ.get("CONSENT_URL", "http://127.0.0.1:8546")
HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "user_app.html")

try:
    from src.features.input_type_filter import classify_input_type
except Exception:  # pragma: no cover
    classify_input_type = None
try:
    from src.features.input_type_filter import MIN_WORDS
except Exception:
    MIN_WORDS = 6

THEMES = {
    "Loss of interest": ["nothing feels worth", "no interest", "don't enjoy", "don't feel like doing", "lost interest", "nothing matters"],
    "Hopelessness": ["no point", "hopeless", "nothing will change", "never get better", "what's the point"],
    "Low self-worth": ["worthless", "burden", "i hate myself", "failure", "useless", "not good enough"],
    "Low energy": ["can't get out of bed", "lie in bed", "so tired", "exhausted", "no energy"],
    "Sleep problems": ["can't sleep", "awake all night", "insomnia", "sleeping all day"],
    "Withdrawal": ["avoid everyone", "stopped talking", "alone all day", "don't see anyone", "pushed everyone away"],
    "Numbness": ["feel numb", "feel empty", "feel nothing", "empty inside"],
    "Thoughts of disappearing": ["disappear", "want to die", "end it all", "not be here", "better off without me"],
}

STATE = {"consent": "none", "share": False, "tx": None, "granted_at": None, "chain_online": False}


# ---------------------------------------------------------------- chain bridge
def chain(route, post=True, timeout=25):
    """Call the Node consent bridge. Returns dict; {'offline': True} if it is not running."""
    try:
        req = urllib.request.Request(CHAIN_URL + route, data=b"{}" if post else None,
                                     method="POST" if post else "GET",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        STATE["chain_online"] = True
        return out
    except Exception as e:  # bridge not running
        STATE["chain_online"] = False
        return {"offline": True, "ok": False, "error": str(e)}


# ---------------------------------------------------------------- robot bridge
class Robot:
    """Thin wrapper over the project's own robot_bus (same calls app.py uses)."""
    def __init__(self, bus, processing, from_rep, error):
        self.bus, self.processing, self.from_rep, self.error = bus, processing, from_rep, error

    def processing_now(self):
        try: self.bus.publish(self.processing())
        except Exception: pass

    def idle(self):
        try: self.bus.publish(self.from_rep({"status": "BLOCKED"}, report_id=0))
        except Exception: pass

    def failed(self):
        try: self.bus.publish(self.error(reason="analysis_exception"))
        except Exception: pass

    def result(self, overall, avg_conf, rid):
        if overall == "red":
            rep = {"status": "AGREEMENT", "risk_class": "depression", "risk_level": "High",
                   "confidence": avg_conf, "meter_pct": round(avg_conf * 100, 1)}
        elif overall == "amber":
            rep = {"status": "DISAGREEMENT"}
        else:
            rep = {"status": "AGREEMENT", "risk_class": "not_depression", "risk_level": "Low",
                   "confidence": avg_conf, "meter_pct": round((1 - avg_conf) * 100, 1)}
        try: self.bus.publish(self.from_rep(rep, report_id=rid))
        except Exception: pass

    def snapshot(self):
        try: return {"ok": True, "data": self.bus.snapshot()}
        except Exception: return {"ok": False}


# ---------------------------------------------------------------- analysis helpers
def find_hits(text):
    low = text.lower()
    themes, hits = [], []
    for name, phrases in THEMES.items():
        for p in phrases:
            if p in low:
                if name not in themes:
                    themes.append(name)
                hits.append(p)
    return themes, hits


def judge_post(text, base_fn, bert_fn, thresholds):
    words = re.findall(r"[\w']+", text)
    if len(words) < MIN_WORDS:
        return {"status": "short", "note": "Too short to check"}
    if classify_input_type:
        it = classify_input_type(text)
        if it.get("category") != "personal_disclosure":
            return {"status": "na", "note": "Not a personal statement (" + it.get("category", "").replace("_", " ") + ")"}
    tb, tr = thresholds
    bp = float(base_fn([text])[0])
    rp = float(bert_fn([text])[0]) if bert_fn else bp
    b_lab = "depression" if bp >= tb else "not_depression"
    r_lab = "depression" if rp >= tr else "not_depression"
    models = {
        "baseline": {"label": b_lab, "conf": round(bp if b_lab == "depression" else 1 - bp, 3), "dep": round(bp, 3)},
        "bert": {"label": r_lab, "conf": round(rp if r_lab == "depression" else 1 - rp, 3), "dep": round(rp, 3)},
    }
    status = "unsure" if b_lab != r_lab else ("flag" if b_lab == "depression" else "ok")
    return {"status": status, "models": models, "avg_dep": round((bp + rp) / 2, 3)}


def register(app, baseline_dep_probs, bert_dep_probs, build_report, REPORTS,
             robot_bus, processing_command, from_report, error_command, thresholds=(0.40, 0.50)):
    robot = Robot(robot_bus, processing_command, from_report, error_command)
    bp = Blueprint("user_api", __name__)

    @app.after_request
    def _cors(resp):  # lets the page also be opened as a file
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @bp.route("/user")
    def page():
        with open(PAGE, encoding="utf-8") as f:
            return Response(f.read(), mimetype="text/html")

    @bp.route("/api/user/health")
    def health():
        c = chain("/status", post=False, timeout=3)
        return jsonify({
            "engine": True,
            "chain": {"online": not c.get("offline"), "address": c.get("address"),
                      "valid": c.get("hasValidConsent")},
            "robot": robot.snapshot(),
            "bertReal": bert_dep_probs is not None,
            "consent": STATE["consent"],
        })

    @bp.route("/api/user/consent", methods=["POST", "OPTIONS"])
    def consent():
        if request.method == "OPTIONS":
            return ("", 204)
        body = request.get_json(silent=True) or {}
        action = body.get("action", "grant")
        if action == "grant":
            res = chain("/grant")
            STATE.update(consent="granted", share=bool(body.get("share")),
                         granted_at=datetime.now().strftime("%d %b %Y, %H:%M"))
        else:
            res = chain("/revoke")
            STATE.update(consent="revoked", share=False)
            robot.idle()
        STATE["tx"] = res.get("txHash")
        return jsonify({"status": "Granted" if action == "grant" else "Revoked",
                        "tx": res.get("txHash"), "block": res.get("block"),
                        "onChain": not res.get("offline"),
                        "note": "Recorded on the local blockchain." if not res.get("offline")
                                else "Blockchain bridge is not running, so this was recorded locally only."})

    @bp.route("/api/user/analyse", methods=["POST", "OPTIONS"])
    def analyse():
        if request.method == "OPTIONS":
            return ("", 204)
        body = request.get_json(silent=True) or {}
        posts = [p for p in body.get("posts", []) if str(p.get("text", "")).strip()]

        def emit(obj):
            return json.dumps(obj) + "\n"

        @stream_with_context
        def stream():
            # ---- consent gate: local state AND the contract
            if STATE["consent"] != "granted":
                yield emit({"type": "blocked", "reason": "No permission has been given."})
                return
            gate = chain("/check")
            if not gate.get("offline") and not gate.get("ok"):
                yield emit({"type": "blocked", "reason": "The consent contract refused processing (permission revoked).",
                            "chain": gate})
                return
            yield emit({"type": "start", "total": len(posts), "gateTx": gate.get("txHash"),
                        "onChain": not gate.get("offline")})
            robot.processing_now()
            results = []
            for i, p in enumerate(posts):
                text = str(p["text"]).strip()
                try:
                    j = judge_post(text, baseline_dep_probs, bert_dep_probs, thresholds)
                except Exception as e:
                    j = {"status": "na", "note": "Could not check: " + str(e)[:80]}
                themes, hits = find_hits(text) if j["status"] in ("flag", "unsure") else ([], [])
                item = {"type": "post", "i": i + 1, "date": p.get("date", ""), "text": text,
                        "themes": themes, "hits": hits, **j}
                results.append(item)
                yield emit(item)
            # ---- overall rule (same as the person report)
            checked = [r for r in results if r["status"] in ("flag", "ok", "unsure")]
            flag = [r for r in results if r["status"] == "flag"]
            unsure = [r for r in results if r["status"] == "unsure"]
            disappear = any("Thoughts of disappearing" in r["themes"] for r in results)
            overall = "green"
            if (len(flag) >= 2 and checked and len(flag) / len(checked) >= 0.25) or disappear:
                overall = "red"
            elif len(flag) >= 1 or len(unsure) >= 2:
                overall = "amber"
            avg = (sum(r["avg_dep"] for r in checked) / len(checked)) if checked else 0.0
            conf = avg if overall == "red" else (1 - avg if overall == "green" else 0.5)
            rid = (len(REPORTS) + 1)
            robot.result(overall, conf, rid)
            shared_id = None
            if STATE["share"] and build_report is not None and REPORTS is not None and (flag or unsure):
                try:
                    top = (flag or unsure)[0]
                    path = "results/blockchain_consent_state_granted.json"
                    rep = build_report(top["text"], "user-shared-report", path)
                    REPORTS.append(rep)
                    shared_id = rep.get("id")
                except Exception:
                    shared_id = None
            yield emit({"type": "done", "overall": overall, "urgent": disappear, "sharedReportId": shared_id,
                        "robot": robot.snapshot()})

        return Response(stream(), mimetype="application/x-ndjson",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    app.register_blueprint(bp)
    print("User app ready at http://127.0.0.1:5000/user  (consent bridge: %s)" % CHAIN_URL)
