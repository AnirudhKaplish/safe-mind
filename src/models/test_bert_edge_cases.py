
"""
test_bert_edge_cases.py
=========================
Runs the same 10-category edge-case suite (and the 5 known hard
cases) through the FULL pipeline: personal-disclosure pre-filter
FIRST, then the calibrated BERT model - exactly how it would run in
production. BERT itself has no idea what a "question" or "third
party" is; that logic lives entirely in input_type_filter.py and
applies identically regardless of which classifier sits behind it.

Usage:
    python src/models/test_bert_edge_cases.py
    python src/models/test_bert_edge_cases.py --file edge_case_tests.txt
"""

import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.bert_inference import predict_one
from src.features.input_type_filter import classify_input_type

BERT_DEPRESSION_THRESHOLD = 0.50  # confirmed optimal by calibrate_bert.py - no adjustment needed


def classify_with_bert(text: str):
    input_type = classify_input_type(text)
    if input_type["category"] != "personal_disclosure":
        print(f"\nINPUT   : {text}")
        print(f"SKIPPED : not a personal disclosure (detected: {input_type['category']})")
        print(f"REASON  : {input_type['reason']}")
        return

    result = predict_one(text)
    dep_prob = result["probabilities"]["depression"]
    pred_class = "depression" if dep_prob >= BERT_DEPRESSION_THRESHOLD else "not_depression"
    confidence = dep_prob if pred_class == "depression" else 1 - dep_prob

    print(f"\nINPUT      : {text}")
    print(f"PREDICTION : {pred_class}")
    print(f"CONFIDENCE : {confidence*100:.1f}%  (calibrated)")


KNOWN_HARD_CASES = [
    "I can't get out of bed, everything feels pointless and I just want to disappear.",
    "I don't even know why I'm posting this. Nothing feels worth doing anymore, I just lie in bed most of the day.",
    "I used to feel depressed but honestly I'm doing so much better now, life is good.",
    "I haven't left my room in three days and I don't see the point in getting up anymore.",
    "This traffic is honestly depressing, I could just scream.",
    "I'm just so tired.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            sentences = [line.strip() for line in f if line.strip()]
    else:
        sentences = KNOWN_HARD_CASES

    print("=" * 70)
    print("RUNNING THROUGH FULL PIPELINE: pre-filter -> calibrated BERT")
    print("=" * 70)
    for s in sentences:
        classify_with_bert(s)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()