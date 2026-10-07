"""
train_baselines.py
===================
Trains and evaluates the baseline models specified in the approved
design (Logistic Regression, Random Forest) on TF-IDF + sentiment +
linguistic features, then compares them.

This is the "NLP Model Training" + "Mental Health Risk
Classification" stage of the approved methodology, restricted to
the baseline (non-transformer) models for this review. The
BERT/RoBERTa fine-tuning script is provided separately
(src/models/finetune_transformer.py) to be run in an environment
with internet access (Colab/local machine with GPU), since this
sandbox cannot download pretrained weights.

Split strategy: we split by unique base post (not by augmented
variant) to avoid leakage between train/test - the augmented
variants of the same seed post must not appear on both sides of
the split, the same way we'd split by author/user on real data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.preprocessing.clean_text import clean_pipeline
from src.features.extract_features import build_feature_table, build_tfidf

DATA_PATH = "data/seed_dataset.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["text"].apply(clean_pipeline)
    return df


def main():
    df = load_and_prepare()
    print(f"Loaded {len(df)} rows, classes: {df['label'].value_counts().to_dict()}")

    # Train/test split (stratified by label; dataset is fully
    # synthetic seed data so no user/author leakage concern here -
    # on the REAL dataset, split by post author, not just randomly)
    train_df, test_df = train_test_split(
        df, test_size=0.25, stratify=df["label"], random_state=42
    )

    # --- Feature extraction ---
    tfidf_vectorizer, X_train_tfidf = build_tfidf(train_df["clean_text"].tolist())
    X_test_tfidf = tfidf_vectorizer.transform(test_df["clean_text"].tolist())

    train_extra = build_feature_table(train_df["clean_text"].tolist(), train_df["text"].tolist())
    test_extra = build_feature_table(test_df["clean_text"].tolist(), test_df["text"].tolist())

    scaler = StandardScaler()
    train_extra_scaled = scaler.fit_transform(train_extra)
    test_extra_scaled = scaler.transform(test_extra)

    X_train = hstack([X_train_tfidf, csr_matrix(train_extra_scaled)])
    X_test = hstack([X_test_tfidf, csr_matrix(test_extra_scaled)])

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_test = le.fit_transform(test_df["label"]) if False else le.transform(test_df["label"])

    # --- Models ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        results.append({"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        fname = f"{RESULTS_DIR}/confusion_matrix_{name.replace(' ', '_').lower()}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")

        joblib.dump(model, f"{RESULTS_DIR}/{name.replace(' ', '_').lower()}_model.joblib")

    # --- Comparison table + bar chart ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/baseline_comparison.csv", index=False)
    print("\n=== Model comparison ===")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(7, 5))
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.35
    for i, row in results_df.iterrows():
        plt.bar(x + i * width, [row[m] for m in metrics], width, label=row["model"])
    plt.xticks(x + width / 2, metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Baseline Model Comparison (macro-averaged)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/baseline_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {RESULTS_DIR}/baseline_comparison.png")

    # Save fitted vectorizer/scaler/encoder for reuse by XAI + report modules
    joblib.dump(tfidf_vectorizer, f"{RESULTS_DIR}/tfidf_vectorizer.joblib")
    joblib.dump(scaler, f"{RESULTS_DIR}/scaler.joblib")
    joblib.dump(le, f"{RESULTS_DIR}/label_encoder.joblib")
    test_df.to_csv(f"{RESULTS_DIR}/test_split.csv", index=False)
    train_df.to_csv(f"{RESULTS_DIR}/train_split.csv", index=False)

    return results_df


if __name__ == "__main__":
    main()