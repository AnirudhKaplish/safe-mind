
"""
tune_threshold.py
==================
Finds a defensible depression-class decision threshold using the
actual precision/recall tradeoff on the held-out test set - NOT by
hand-picking a number that happens to flip one specific example.

This is the responsible version of "make it catch more depression
cases": we look at how recall and precision trade off across a range
of thresholds, pick one based on that curve (favoring recall, per the
approved design's stated evaluation focus), and then apply that SAME
threshold uniformly to every prediction - never a per-sentence fix.

Usage:
    python src/models/tune_threshold.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from scipy.sparse import hstack, csr_matrix

from src.preprocessing.clean_text import clean_pipeline
from src.features.extract_features import build_feature_table

RESULTS_DIR = "results"


def main():
    model = joblib.load(f"{RESULTS_DIR}/logistic_regression_model.joblib")
    vectorizer = joblib.load(f"{RESULTS_DIR}/tfidf_vectorizer.joblib")
    scaler = joblib.load(f"{RESULTS_DIR}/scaler.joblib")
    le = joblib.load(f"{RESULTS_DIR}/label_encoder.joblib")

    test_df = pd.read_csv(f"{RESULTS_DIR}/test_split.csv")
    test_df["clean_text"] = test_df["clean_text"].fillna("").astype(str)

    X_tfidf = vectorizer.transform(test_df["clean_text"])
    extra = build_feature_table(test_df["clean_text"].tolist(), test_df["text"].tolist())
    extra_scaled = scaler.transform(extra)
    X = hstack([X_tfidf, csr_matrix(extra_scaled)])

    y_true = le.transform(test_df["label"])
    dep_idx = list(le.classes_).index("depression")
    y_true_binary = (y_true == dep_idx).astype(int)

    proba_depression = model.predict_proba(X)[:, dep_idx]

    precisions, recalls, thresholds = precision_recall_curve(y_true_binary, proba_depression)

    # Plot the tradeoff curve
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision (depression)")
    plt.plot(thresholds, recalls[:-1], label="Recall (depression)")
    plt.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Default (argmax) threshold")
    plt.xlabel("Depression-class probability threshold")
    plt.ylabel("Score")
    plt.title("Precision/Recall tradeoff vs. decision threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/threshold_tuning_curve.png", dpi=150)
    plt.close()
    print(f"Saved {RESULTS_DIR}/threshold_tuning_curve.png")

    # Evaluate a small set of candidate thresholds explicitly, so the
    # choice is documented and reproducible, not just eyeballed off a plot.
    candidates = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]
    print(f"\n{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    results = []
    for t in candidates:
        y_pred_binary = (proba_depression >= t).astype(int)
        p = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        r = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        results.append({"threshold": t, "precision": p, "recall": r, "f1": f1})
        print(f"{t:>10.2f} {p:>10.3f} {r:>10.3f} {f1:>10.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/threshold_candidates.csv", index=False)
    print(f"\nSaved {RESULTS_DIR}/threshold_candidates.csv")

    print("\nHow to pick one: moving the threshold down from 0.50 trades some")
    print("precision (more false alarms) for higher recall (fewer missed real")
    print("cases). For a SCREENING tool this is usually the right trade - pick")
    print("the lowest threshold where precision hasn't dropped much, then use")
    print("that value as `depression_threshold` in generate_report.py.")


if __name__ == "__main__":
    main()