
"""
test_my_own_sentences.py
=========================
Interactive tester: type in your own sentence, see what the model
predicts, how confident it is, and WHY (via LIME word-level evidence).

This is the most honest way to sanity-check the model - not just
trusting an accuracy percentage, but actually reading real predictions
and judging for yourself whether they look right.

Usage:
    python test_my_own_sentences.py

Then just type sentences, one per line. Type 'quit' to exit.
You can also pass sentences as a batch from a file:
    python test_my_own_sentences.py --file my_test_sentences.txt
"""

import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.xai.explain import load_artifacts, featurize, generate_explanation_report
from src.features.input_type_filter import classify_input_type

def classify_and_explain(text, depression_threshold=0.40):
    input_type = classify_input_type(text)
    if input_type["category"] != "personal_disclosure":
        print("\n" + "=" * 65)
        print(f"INPUT      : {text}")
        print(f"SKIPPED    : not a personal disclosure (detected: {input_type['category']})")
        print(f"REASON     : {input_type['reason']}")
        print("=" * 65)
        return

    model, vectorizer, scaler, le = load_artifacts()
    X = featurize([text], vectorizer, scaler)
    proba = model.predict_proba(X)[0]

    # Same recall-favoring decision rule as generate_report.py -
    # flag depression whenever it clears depression_threshold, not
    # just when it's the single highest-probability class. Chosen
    # from a real precision/recall tradeoff analysis
    # (src/models/tune_threshold.py), not hand-tuned per sentence.
    depression_idx = list(le.classes_).index("depression") if "depression" in le.classes_ else None
    if depression_idx is not None and proba[depression_idx] >= depression_threshold:
        pred_idx = depression_idx
    else:
        pred_idx = int(proba.argmax())
    pred_class = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    print("\n" + "=" * 65)
    print(f"INPUT      : {text}")
    print(f"PREDICTION : {pred_class}")
    print(f"CONFIDENCE : {confidence*100:.1f}%")
    if confidence < 0.65:
        print("  ^ LOW CONFIDENCE - treat this prediction as uncertain/borderline")

    try:
        explanation = generate_explanation_report(text)
        print("\nTop words driving this prediction (LIME):")
        for word, val in explanation["lime"]["top_words"][:6]:
            direction = "supports" if val > 0 else "argues against"
            print(f"   {word:20s}  ({direction} '{pred_class}',  weight {val:+.3f})")
    except Exception as e:
        print(f"(explanation unavailable: {e})")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                         help="Optional: path to a text file, one sentence per line")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            sentences = [line.strip() for line in f if line.strip()]
        for s in sentences:
            classify_and_explain(s)
        return

    print("Type a sentence to classify (or 'quit' to exit):\n")
    while True:
        try:
            text = input(">> ").strip()
        except EOFError:
            break
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        classify_and_explain(text)


if __name__ == "__main__":
    main()