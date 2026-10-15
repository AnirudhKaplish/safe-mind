"""Minimal stand-in for src/dashboard/app.py, used ONLY when the real
dashboard is not importable (e.g. CI outside the full repo).

It reproduces the request flow documented in the build prompt (section 1)
with exactly the robot edits of section 10 applied, so the wiring can be
tested. When the real app exists, the integration tests use the real app.
"""
from flask import Flask, jsonify, redirect, request

from robot_sim.bus import robot_bus
from robot_sim.controller import error_command, from_report, processing_command

app = Flask(__name__)
REPORTS = []
CONSENT_GRANTED = "results/blockchain_consent_state_granted.json"
CONSENT_DEMO = "results/blockchain_consent_demo_log.json"


def classify_input_type(text):
    t = text.strip().lower()
    personal = (t.startswith("i ") or " i " in f" {t} " or " my " in f" {t} ") and not t.endswith("?")
    return {"is_disclosure": personal, "category": "personal" if personal else "question"}


def baseline_predict(text):  # replaced by tests
    raise RuntimeError("model not available in reference app")


def bert_predict(text):  # replaced by tests
    raise RuntimeError("model not available in reference app")


def get_consent_status(path):
    return "granted" if path == CONSENT_GRANTED else "revoked"


def build_report(text, patient_id, consent_log_path):
    kind = classify_input_type(text)
    if not kind["is_disclosure"]:
        return {"status": "NOT_APPLICABLE", "reason": f"not a disclosure: {text[:30]}",
                "category": kind["category"], "model_breakdown": [], "consent_status": "n/a"}
    consent = get_consent_status(consent_log_path)
    if consent != "granted":
        return {"status": "BLOCKED", "consent_status": consent, "model_breakdown": []}
    b, m = baseline_predict(text), bert_predict(text)
    breakdown = [b, m]
    if b["prediction"] == m["prediction"]:
        conf = (b["confidence"] + m["confidence"]) / 2
        return {"status": "AGREEMENT", "risk_class": b["prediction"],
                "risk_level": "High" if b["prediction"] == "depression" else "Low",
                "confidence": conf, "meter_pct": int(conf * 100), "model_breakdown": breakdown,
                "consent_status": consent, "patient_id": patient_id}
    return {"status": "DISAGREEMENT", "model_breakdown": breakdown, "consent_status": consent,
            "patient_id": patient_id}


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "")
    patient_id = request.form.get("patient_id", "")
    scenario = request.form.get("consent_scenario", "granted")
    if not text.strip():
        return redirect("/")
    consent_log_path = CONSENT_GRANTED if scenario == "granted" else CONSENT_DEMO
    robot_bus.publish(processing_command())
    try:
        report = build_report(text, patient_id, consent_log_path)
    except Exception:
        robot_bus.publish(error_command(reason="analysis_exception"))
        raise
    report_id = len(REPORTS)
    REPORTS.append(report)
    robot_bus.publish(from_report(report, report_id=report_id))
    return redirect(f"/report/{report_id}")


@app.route("/api/robot/state")
def robot_state():  # read-only
    resp = jsonify(robot_bus.snapshot())
    resp.headers["Cache-Control"] = "no-store"
    return resp
