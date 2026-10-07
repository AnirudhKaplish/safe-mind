
"""
calibrate_bert.py
==================
Two things, both standard, well-established ML practices:

1. TEMPERATURE SCALING (Guo et al. 2017, "On Calibration of Modern
   Neural Networks"): fine-tuned transformers are well-documented to
   be overconfident - exactly what we saw (99%+ confidence on WRONG
   answers). Temperature scaling fixes this with a single learned
   scalar T: calibrated_probs = softmax(logits / T). T > 1 makes the
   model less confident/more honest; T=1 means no change. This does
   NOT change which class wins (argmax is unaffected) - it only
   makes the confidence NUMBER trustworthy.

2. DECISION THRESHOLD TUNING: same recall-favoring logic already
   applied to the baseline model (src/models/tune_threshold.py),
   now applied to BERT's (calibrated) depression probability.

Run this ONCE after training/downloading your BERT model, using the
SAME dataset + same random_state split used during Colab training,
so the "test" rows here were genuinely held out during fine-tuning
(not seen during training) - this is essential for a valid
calibration, not something we're just re-checking for style.

Usage:
    python src/models/calibrate_bert.py --data data/augmented_dataset.csv
"""

import argparse
import json
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "results/bert_model"
RESULTS_DIR = "results"


def get_raw_logits(texts, tokenizer, model, batch_size=16):
    """Runs the model over all texts and returns raw (uncalibrated) logits."""
    all_logits = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        all_logits.append(logits)
        if i % (batch_size * 10) == 0:
            print(f"  scored {i}/{len(texts)}...")
    return torch.cat(all_logits, dim=0)


def find_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter=200) -> float:
    """Learns the single scalar T that minimizes negative log-likelihood
    of softmax(logits / T) against true labels - standard temperature
    scaling optimization using LBFGS (per Guo et al. 2017)."""
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    nll_criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.item())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins=10) -> float:
    """ECE: bins predictions by confidence, compares average confidence
    to actual accuracy within each bin. Lower is better (0 = perfectly
    calibrated). This is THE standard metric for 'is this model's
    confidence trustworthy', separate from whether it's ACCURATE."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)
    return ece


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/augmented_dataset.csv")
    args = parser.parse_args()

    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    classes = list(le.classes_)
    print("Classes:", classes)

    # SAME split logic/seed as the Colab training script - reproduces
    # the exact held-out test set that was NOT used for training.
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)
    print(f"Held-out test set: {len(test_df)} rows")

    print("Scoring held-out test set (this runs on CPU, may take a few minutes)...")
    logits = get_raw_logits(test_df["text"].tolist(), tokenizer, model)
    labels = torch.tensor(test_df["label_id"].tolist())

    # --- Before calibration ---
    raw_probs = torch.softmax(logits, dim=-1).numpy()
    ece_before = expected_calibration_error(raw_probs, labels.numpy())
    print(f"\nExpected Calibration Error BEFORE temperature scaling: {ece_before:.4f}")

    # --- Find and apply temperature ---
    temperature = find_temperature(logits, labels)
    print(f"Learned temperature: {temperature:.3f}")

    calibrated_probs = torch.softmax(logits / temperature, dim=-1).numpy()
    ece_after = expected_calibration_error(calibrated_probs, labels.numpy())
    print(f"Expected Calibration Error AFTER temperature scaling: {ece_after:.4f}")

    with open(f"{RESULTS_DIR}/bert_temperature.json", "w") as f:
        json.dump({"temperature": temperature, "ece_before": ece_before, "ece_after": ece_after}, f, indent=2)
    print(f"Saved {RESULTS_DIR}/bert_temperature.json")

    # --- Reliability diagram (before vs after) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, probs, title in [(axes[0], raw_probs, "Before calibration"),
                              (axes[1], calibrated_probs, "After calibration")]:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels.numpy()).astype(float)
        bins = np.linspace(0, 1, 11)
        bin_acc, bin_conf = [], []
        for i in range(10):
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
            if mask.sum() > 0:
                bin_acc.append(accuracies[mask].mean())
                bin_conf.append(confidences[mask].mean())
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.scatter(bin_conf, bin_acc, color="crimson")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Actual accuracy")
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/bert_calibration_curve.png", dpi=150)
    plt.close()
    print(f"Saved {RESULTS_DIR}/bert_calibration_curve.png")

    # --- Threshold tuning on CALIBRATED depression probability ---
    dep_idx = classes.index("depression")
    y_true_binary = (test_df["label_id"] == dep_idx).astype(int).to_numpy()
    proba_depression = calibrated_probs[:, dep_idx]

    candidates = [0.50, 0.45, 0.40, 0.35, 0.30]
    print(f"\n{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    results = []
    for t in candidates:
        y_pred_binary = (proba_depression >= t).astype(int)
        p = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        r = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        results.append({"threshold": t, "precision": p, "recall": r, "f1": f1})
        print(f"{t:>10.2f} {p:>10.3f} {r:>10.3f} {f1:>10.3f}")

    pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/bert_threshold_candidates.csv", index=False)
    print(f"\nSaved {RESULTS_DIR}/bert_threshold_candidates.csv")
    print("\nPick a threshold from the table above (same logic as the baseline:")
    print("lower threshold = more recall, less precision) and set it as")
    print("BERT_DEPRESSION_THRESHOLD wherever BERT is used for classification.")


if __name__ == "__main__":
    main()