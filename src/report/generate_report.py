"""
generate_report.py
==================
Baseline (Logistic Regression + LIME) assessment report with blockchain
consent enforcement. This is the module that run_all.sh and
hardware/virtual_device.py import. It needs only the baseline model
artifacts in results/ (no BERT model required).

Run with:
    python src/report/generate_report.py
"""

import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime, timezone

from src.xai.explain import load_artifacts, featurize, explain_with_lime
from src.features.input_type_filter import classify_input_type

RESULTS_DIR = "results"
DEPRESSION_THRESHOLD = 0.40  # same recall-favouring rule as test_my_own_sentences.py
RISK_LEVEL_MAP = {"not_depression": "Low", "depression": "High"}
REPORT_TYPE = "Mental-Health Assessment Report (Baseline)"


def get_consent_status(consent_log_path=f"{RESULTS_DIR}/blockchain_consent_demo_log.json"):
    if not os.path.exists(consent_log_path):
        return {"status": "unknown", "note": "no consent log found - run blockchain demo first"}
    with open(consent_log_path) as f:
        log = json.load(f)
    steps = log.get("steps", [])
    return {
        "status": "revoked" if log.get("blockedConfirmed") else "granted",
        "contract_address": log.get("contractAddress"),
        "last_action": steps[-2]["step"] if len(steps) >= 2 else None,
    }


def generate_assessment_report(text: str, patient_id: str = "demo-patient-001",
                               override_consent_check: bool = False,
                               consent_log_path: str = f"{RESULTS_DIR}/blockchain_consent_demo_log.json") -> dict:
    consent = get_consent_status(consent_log_path)
    base = {
        "report_type": REPORT_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patient_id": patient_id,
        "consent_status": consent,
    }

    input_type = classify_input_type(text)
    if input_type["category"] != "personal_disclosure":
        return {**base, "status": "NOT_APPLICABLE",
                "reason": f"Input does not appear to be a first-person personal disclosure "
                          f"(detected: {input_type['category']} - {input_type['reason']}). "
                          f"Risk classification was not run."}

    if consent["status"] != "granted" and not override_consent_check:
        return {**base, "status": "BLOCKED",
                "reason": f"Processing blocked - consent status is '{consent['status']}', not 'granted'."}

    model, vectorizer, scaler, le = load_artifacts()
    proba = model.predict_proba(featurize([text], vectorizer, scaler))[0]
    classes = list(le.classes_)
    if "depression" in classes and proba[classes.index("depression")] >= DEPRESSION_THRESHOLD:
        pred_idx = classes.index("depression")
    else:
        pred_idx = int(proba.argmax())
    pred_class = classes[pred_idx]

    lime = explain_with_lime(text, model, vectorizer, scaler, le)

    return {**base, "status": "ALLOWED",
            "risk_class": pred_class,
            "risk_level": RISK_LEVEL_MAP.get(pred_class, "Unknown"),
            "confidence": round(float(proba[pred_idx]), 4),
            "evidence": {"lime_top_words": [[str(w), round(float(v), 4)] for w, v in lime["top_words"]]},
            "intended_user": "Doctor / Therapist",
            "purpose": "Decision support for further human-led assessment",
            "diagnosis_note": "This report does NOT constitute an autonomous diagnosis. Final clinical "
                              "judgement rests with the reviewing doctor/therapist."}


if __name__ == "__main__":
    sample = ("I don't even know why I'm posting this. Nothing feels worth doing anymore, "
              "I just lie in bed most of the day.")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    allowed = generate_assessment_report(
        sample, consent_log_path=f"{RESULTS_DIR}/blockchain_consent_state_granted.json")
    blocked = generate_assessment_report(
        sample, consent_log_path=f"{RESULTS_DIR}/blockchain_consent_demo_log.json")

    for name, rep in [("allowed", allowed), ("blocked", blocked)]:
        path = f"{RESULTS_DIR}/sample_assessment_report_{name}.json"
        with open(path, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\n[{name.upper()} scenario] status = {rep['status']}")
        if rep["status"] == "ALLOWED":
            print(f"  risk_class = {rep['risk_class']}  confidence = {rep['confidence']:.2f}")
            print(f"  top words  = {[w for w, _ in rep['evidence']['lime_top_words'][:5]]}")
        else:
            print(f"  reason     = {rep['reason']}")
        print(f"  saved -> {path}")
