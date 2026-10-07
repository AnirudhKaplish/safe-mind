"""
generate_report_bert.py
=========================
Same "Mental-Health Assessment Report" structure and same pipeline
stages as src/report/generate_report.py (personal-disclosure
pre-filter -> consent check -> classification -> explanation ->
structured report), but backed by the fine-tuned, counterfactually
augmented, temperature-calibrated BERT model instead of the TF-IDF
baseline.

Kept as a SEPARATE, fully self-contained file from generate_report.py
(rather than adding a "use_bert=True" flag, and rather than importing
shared helpers across files) so both pipelines can be run side-by-side
and compared directly - useful for the panel review - without any
cross-module import path issues.

Decision threshold: 0.50 (BERT's default argmax). Unlike the
baseline, calibrate_bert.py's threshold-tuning table showed 0.50 was
ALREADY the best-performing threshold for BERT - no adjustment
needed. This itself is a useful, documented finding: threshold
tuning was compensating for the baseline's specific weaknesses, not
a universal requirement.

Explanation method: LIME only. SHAP's LinearExplainer (used for the
baseline) only works cleanly on linear models - explaining a
transformer properly needs shap.Explainer with a text-classification
pipeline wrapper, which is a heavier, separate piece of work. LIME is
model-agnostic (it just needs a predict_proba function) so it works
here unchanged. Adding SHAP-for-BERT support is a documented next
step, not a silent omission.
"""

import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from datetime import datetime, timezone
from lime.lime_text import LimeTextExplainer

from src.models.bert_inference import predict_proba, load_bert
from src.features.input_type_filter import classify_input_type

RESULTS_DIR = "results"
BERT_DEPRESSION_THRESHOLD = 0.50  # confirmed optimal by calibrate_bert.py

RISK_LEVEL_MAP = {"not_depression": "Low", "depression": "High"}


def get_consent_status(consent_log_path=f"{RESULTS_DIR}/blockchain_consent_demo_log.json"):
    """Reads the most recent on-chain consent state produced by the
    blockchain demo script. Duplicated here (rather than imported from
    generate_report.py) to keep this file fully self-contained and
    avoid cross-module import path issues."""
    if not os.path.exists(consent_log_path):
        return {"status": "unknown", "note": "no consent log found - run blockchain demo first"}
    with open(consent_log_path) as f:
        log = json.load(f)
    return {
        "status": "revoked" if log.get("blockedConfirmed") else "granted",
        "contract_address": log.get("contractAddress"),
        "last_action": log["steps"][-2]["step"] if len(log["steps"]) >= 2 else None,
    }


def _predict_proba_array(texts: list) -> np.ndarray:
    """Wraps bert_inference.predict_proba (list of dicts) into the
    fixed-column ndarray shape LIME expects."""
    _, _, classes, _ = load_bert()
    dict_list = predict_proba(texts)
    return np.array([[d[c] for c in classes] for d in dict_list])


def explain_with_lime_bert(text: str, num_features: int = 8) -> dict:
    _, _, classes, _ = load_bert()
    explainer = LimeTextExplainer(class_names=classes)
    exp = explainer.explain_instance(
        text, _predict_proba_array, num_features=num_features,
        labels=list(range(len(classes))), num_samples=300
    )
    proba = _predict_proba_array([text])[0]
    pred_idx = int(np.argmax(proba))
    pred_class = classes[pred_idx]
    return {
        "predicted_class": pred_class,
        "top_words": exp.as_list(label=pred_idx),
    }


def generate_assessment_report_bert(text: str, patient_id: str = "demo-patient-001",
                                     override_consent_check: bool = False,
                                     consent_log_path: str = f"{RESULTS_DIR}/blockchain_consent_demo_log.json") -> dict:
    consent = get_consent_status(consent_log_path)

    # --- Personal-disclosure pre-filter (identical to baseline) ---
    input_type = classify_input_type(text)
    if input_type["category"] != "personal_disclosure":
        return {
            "report_type": "Mental-Health Assessment Report (BERT)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "patient_id": patient_id,
            "status": "NOT_APPLICABLE",
            "reason": f"Input does not appear to be a first-person personal disclosure "
                      f"(detected: {input_type['category']} - {input_type['reason']}). "
                      f"Risk classification was not run.",
            "consent_status": consent,
        }

    # --- Consent enforcement (identical to baseline) ---
    if consent["status"] != "granted" and not override_consent_check:
        return {
            "report_type": "Mental-Health Assessment Report (BERT)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "patient_id": patient_id,
            "status": "BLOCKED",
            "reason": f"Processing blocked - consent status is '{consent['status']}', not 'granted'.",
            "consent_status": consent,
        }

    # --- Calibrated BERT classification ---
    dict_list = predict_proba([text])
    proba_dict = dict_list[0]
    dep_prob = proba_dict["depression"]
    pred_class = "depression" if dep_prob >= BERT_DEPRESSION_THRESHOLD else "not_depression"
    confidence = dep_prob if pred_class == "depression" else 1 - dep_prob

    lime_result = explain_with_lime_bert(text)

    report = {
        "report_type": "Mental-Health Assessment Report (BERT)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patient_id": patient_id,
        "risk_class": pred_class,
        "risk_level": RISK_LEVEL_MAP.get(pred_class, "Unknown"),
        "confidence": round(float(confidence), 4),
        "confidence_calibrated": True,
        "indicators": pred_class,
        "evidence": {
            "lime_top_words": lime_result["top_words"],
        },
        "explanation_method": "LIME (LimeTextExplainer) on calibrated BERT - SHAP-for-transformers not yet implemented",
        "consent_status": consent,
        "intended_user": "Doctor / Therapist",
        "purpose": "Decision support for further human-led assessment",
        "diagnosis_note": "This report does NOT constitute an autonomous diagnosis. "
                           "Final clinical judgement rests with the reviewing doctor/therapist.",
    }
    return report


def render_human_readable(report: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(report["report_type"])
    lines.append("=" * 70)
    lines.append(f"Patient ID       : {report['patient_id']}")
    lines.append(f"Generated at     : {report['generated_at']}")
    lines.append(f"Risk Class       : {report['risk_class']}")
    lines.append(f"Risk Level       : {report['risk_level']}")
    lines.append(f"Confidence       : {report['confidence']*100:.1f}% (calibrated)")
    lines.append(f"Consent status   : {report['consent_status']['status']}")
    lines.append("")
    lines.append("Top contributing words (LIME):")
    for word, val in report["evidence"]["lime_top_words"][:5]:
        sign = "+" if val > 0 else "-"
        lines.append(f"   {sign} {word} ({val:+.3f})")
    lines.append("")
    lines.append(f"Purpose: {report['purpose']}")
    lines.append(f"NOTE: {report['diagnosis_note']}")
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    sample_text = "I don't even know why I'm posting this. Nothing feels worth doing anymore, I just lie in bed most of the day."

    print("\n" + "#" * 70)
    print("# SCENARIO A: consent currently GRANTED -> report should generate")
    print("#" * 70)
    report_allowed = generate_assessment_report_bert(
        sample_text, consent_log_path=f"{RESULTS_DIR}/blockchain_consent_state_granted.json"
    )
    with open(f"{RESULTS_DIR}/sample_assessment_report_bert_allowed.json", "w") as f:
        json.dump(report_allowed, f, indent=2)
    if report_allowed.get("status") in ("BLOCKED", "NOT_APPLICABLE"):
        print(json.dumps(report_allowed, indent=2))
    else:
        print(render_human_readable(report_allowed))

    print("\n" + "#" * 70)
    print("# SCENARIO B: consent currently REVOKED -> report should be BLOCKED")
    print("#" * 70)
    report_blocked = generate_assessment_report_bert(
        sample_text, consent_log_path=f"{RESULTS_DIR}/blockchain_consent_demo_log.json"
    )
    with open(f"{RESULTS_DIR}/sample_assessment_report_bert_blocked.json", "w") as f:
        json.dump(report_blocked, f, indent=2)
    print(json.dumps(report_blocked, indent=2))