
"""
generate_report_ensemble.py
=============================
Runs BOTH the baseline (TF-IDF + LR) and the fine-tuned, calibrated
BERT model on the same input. If they AGREE, that agreement itself is
extra evidence the call is solid. If they DISAGREE, the ensemble does
NOT pick a winner - it flags the case for human review instead. This
directly matches the project's own design principle (decision support,
never autonomous diagnosis): when two independent models can't agree,
neither should be trusted blindly.

Fully self-contained (duplicates the small consent-check helper and
calls src.xai.explain / src.models.bert_inference / src.features.
input_type_filter directly) to avoid any cross-module import path
issues between files inside src/report/.
"""

import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime, timezone

from src.xai.explain import load_artifacts, featurize
from src.models.bert_inference import predict_proba as bert_predict_proba
from src.features.input_type_filter import classify_input_type

RESULTS_DIR = "results"
BASELINE_DEPRESSION_THRESHOLD = 0.40
BERT_DEPRESSION_THRESHOLD = 0.50

RISK_LEVEL_MAP = {"not_depression": "Low", "depression": "High"}


def get_consent_status(consent_log_path=f"{RESULTS_DIR}/blockchain_consent_demo_log.json"):
    if not os.path.exists(consent_log_path):
        return {"status": "unknown", "note": "no consent log found - run blockchain demo first"}
    with open(consent_log_path) as f:
        log = json.load(f)
    return {
        "status": "revoked" if log.get("blockedConfirmed") else "granted",
        "contract_address": log.get("contractAddress"),
        "last_action": log["steps"][-2]["step"] if len(log["steps"]) >= 2 else None,
    }


def _baseline_predict(text: str) -> dict:
    model, vectorizer, scaler, le = load_artifacts()
    X = featurize([text], vectorizer, scaler)
    proba = model.predict_proba(X)[0]
    depression_idx = list(le.classes_).index("depression")
    if proba[depression_idx] >= BASELINE_DEPRESSION_THRESHOLD:
        pred_idx = depression_idx
    else:
        pred_idx = int(proba.argmax())
    pred_class = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])
    return {"model": "baseline (TF-IDF + Logistic Regression)",
            "prediction": pred_class, "confidence": confidence}


def _bert_predict(text: str) -> dict:
    proba_dict = bert_predict_proba([text])[0]
    dep_prob = proba_dict["depression"]
    pred_class = "depression" if dep_prob >= BERT_DEPRESSION_THRESHOLD else "not_depression"
    confidence = dep_prob if pred_class == "depression" else 1 - dep_prob
    return {"model": "BERT (fine-tuned, calibrated)",
            "prediction": pred_class, "confidence": float(confidence)}


def generate_assessment_report_ensemble(text: str, patient_id: str = "demo-patient-001",
                                         override_consent_check: bool = False,
                                         consent_log_path: str = f"{RESULTS_DIR}/blockchain_consent_demo_log.json") -> dict:
    consent = get_consent_status(consent_log_path)

    # --- Personal-disclosure pre-filter (shared, model-agnostic) ---
    input_type = classify_input_type(text)
    if input_type["category"] != "personal_disclosure":
        return {
            "report_type": "Mental-Health Assessment Report (Ensemble)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "patient_id": patient_id,
            "status": "NOT_APPLICABLE",
            "reason": f"Input does not appear to be a first-person personal disclosure "
                      f"(detected: {input_type['category']} - {input_type['reason']}). "
                      f"Risk classification was not run.",
            "consent_status": consent,
        }

    # --- Consent enforcement ---
    if consent["status"] != "granted" and not override_consent_check:
        return {
            "report_type": "Mental-Health Assessment Report (Ensemble)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "patient_id": patient_id,
            "status": "BLOCKED",
            "reason": f"Processing blocked - consent status is '{consent['status']}', not 'granted'.",
            "consent_status": consent,
        }

    # --- Run both models ---
    baseline_result = _baseline_predict(text)
    bert_result = _bert_predict(text)

    agree = baseline_result["prediction"] == bert_result["prediction"]

    if agree:
        pred_class = baseline_result["prediction"]
        # simple, conservative combined confidence: average of the two,
        # not the max - don't let one overconfident model inflate trust
        combined_confidence = (baseline_result["confidence"] + bert_result["confidence"]) / 2
        report = {
            "report_type": "Mental-Health Assessment Report (Ensemble)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "patient_id": patient_id,
            "status": "AGREEMENT",
            "risk_class": pred_class,
            "risk_level": RISK_LEVEL_MAP.get(pred_class, "Unknown"),
            "confidence": round(combined_confidence, 4),
            "model_breakdown": [baseline_result, bert_result],
            "consent_status": consent,
            "intended_user": "Doctor / Therapist",
            "purpose": "Decision support for further human-led assessment",
            "diagnosis_note": "Both models independently agree on this classification. This report does NOT "
                               "constitute an autonomous diagnosis. Final clinical judgement rests with the "
                               "reviewing doctor/therapist.",
        }
    else:
        report = {
            "report_type": "Mental-Health Assessment Report (Ensemble)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "patient_id": patient_id,
            "status": "DISAGREEMENT_NEEDS_REVIEW",
            "risk_class": None,
            "risk_level": "UNRESOLVED - see model_breakdown",
            "model_breakdown": [baseline_result, bert_result],
            "consent_status": consent,
            "intended_user": "Doctor / Therapist",
            "purpose": "Decision support for further human-led assessment",
            "diagnosis_note": "The baseline and BERT models DISAGREE on this classification. No automatic "
                               "verdict is given - this case is flagged for priority human review rather than "
                               "resolved by picking either model's answer.",
        }
    return report


def render_human_readable(report: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(report["report_type"])
    lines.append("=" * 70)
    lines.append(f"Patient ID       : {report['patient_id']}")
    lines.append(f"Generated at     : {report['generated_at']}")
    lines.append(f"Status           : {report['status']}")
    if report["status"] == "AGREEMENT":
        lines.append(f"Risk Class       : {report['risk_class']}")
        lines.append(f"Risk Level       : {report['risk_level']}")
        lines.append(f"Combined conf.   : {report['confidence']*100:.1f}%")
    lines.append(f"Consent status   : {report['consent_status']['status']}")
    if "model_breakdown" not in report:
        lines.append(f"Reason           : {report.get('reason', '-')}")
        lines.append("=" * 70)
        return "\n".join(lines)
    lines.append("")
    lines.append("Model breakdown:")
    for m in report["model_breakdown"]:
        lines.append(f"   {m['model']:40s} -> {m['prediction']:15s} ({m['confidence']*100:.1f}%)")
    lines.append("")
    lines.append(f"NOTE: {report['diagnosis_note']}")
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    test_cases = [
        "I don't even know why I'm posting this. Nothing feels worth doing anymore, I just lie in bed most of the day.",
        "Had a great time at the party last night with my friends.",
        "I'm just so tired.",
         "This job is genuinely soul-crushing, I don't know how much more of this I can take.",
        "Ever since the surgery I just don't feel like myself anymore, everything's different now.",
    ]
    for text in test_cases:
        print(f"\nINPUT: {text}")
        report = generate_assessment_report_ensemble(
            text, consent_log_path=f"{RESULTS_DIR}/blockchain_consent_state_granted.json"
        )
        if report.get("status") in ("BLOCKED", "NOT_APPLICABLE"):
            print(json.dumps(report, indent=2))
        else:
            print(render_human_readable(report))