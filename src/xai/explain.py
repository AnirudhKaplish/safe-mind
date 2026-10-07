"""
explain.py
==========
Explainable AI module - implements "Explainable AI Analysis" from
the approved methodology: applies SHAP and LIME to the trained
Logistic Regression baseline model to identify which words drove
a given risk-classification prediction.

We explain Logistic Regression here (not Random Forest) because:
  1. LR's linear decision boundary makes SHAP's LinearExplainer
     exact and fast (no sampling approximation needed).
  2. It gives cleanly signed, per-word contribution values which
     are easy to display to a non-technical reviewer (a doctor).
The same approach applies to the transformer model later via
shap.Explainer(model) / lime_text once BERT/RoBERTa is trained.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd  # noqa: F401 (used for background CSV load)
import joblib
from scipy.sparse import hstack, csr_matrix
import shap
from lime.lime_text import LimeTextExplainer

from src.preprocessing.clean_text import clean_pipeline
from src.features.extract_features import build_feature_table

RESULTS_DIR = "results"


def load_artifacts():
    model = joblib.load(f"{RESULTS_DIR}/logistic_regression_model.joblib")
    vectorizer = joblib.load(f"{RESULTS_DIR}/tfidf_vectorizer.joblib")
    scaler = joblib.load(f"{RESULTS_DIR}/scaler.joblib")
    le = joblib.load(f"{RESULTS_DIR}/label_encoder.joblib")
    return model, vectorizer, scaler, le


def featurize(texts, vectorizer, scaler):
    cleaned = [clean_pipeline(t) for t in texts]
    tfidf = vectorizer.transform(cleaned)
    extra = build_feature_table(cleaned, texts)
    extra_scaled = scaler.transform(extra)
    return hstack([tfidf, csr_matrix(extra_scaled)])


_BACKGROUND_CACHE = {}


def _get_background(vectorizer, scaler):
    """SHAP needs a reference background distribution to measure each
    feature's DEVIATION against - explaining a single point against
    itself as its own background trivially gives all-zero contributions.
    We build the background once from the training split and cache it."""
    key = id(vectorizer)
    if key not in _BACKGROUND_CACHE:
        train_df = pd.read_csv(f"{RESULTS_DIR}/train_split.csv")
        bg_texts = train_df["clean_text"].fillna("").astype(str).tolist()
        bg_raw_texts = train_df["text"].fillna("").astype(str).tolist()
        bg_tfidf = vectorizer.transform(bg_texts)
        bg_extra = build_feature_table(bg_texts, bg_raw_texts)
        bg_extra_scaled = scaler.transform(bg_extra)
        bg = hstack([bg_tfidf, csr_matrix(bg_extra_scaled)])
        # shap.LinearExplainer background must be dense-ish / small - sample up to 50 rows
        n = min(50, bg.shape[0])
        idx = np.random.RandomState(42).choice(bg.shape[0], n, replace=False)
        _BACKGROUND_CACHE[key] = bg[idx].toarray()
    return _BACKGROUND_CACHE[key]


def explain_with_shap(text, model, vectorizer, scaler, le, top_k=8):
    """SHAP explanation using LinearExplainer (exact for linear models).
    Explains the instance's prediction relative to a background sample
    drawn from the TRAINING data, not against itself."""
    X = featurize([text], vectorizer, scaler)
    pred_class_idx = model.predict(X)[0]
    pred_class = le.inverse_transform([pred_class_idx])[0]

    feature_names = list(vectorizer.get_feature_names_out()) + \
        ["sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment_compound",
         "word_count", "first_person_ratio", "negation_count",
         "avg_word_len", "exclam_count", "question_count",
         "depression_phrase_count", "depression_phrase_present"]

    background = _get_background(vectorizer, scaler)
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(X.toarray())

    # shap_values shape: (n_samples, n_features, n_classes) for multi-class LinearExplainer
    if isinstance(shap_values, list):
        class_shap = shap_values[pred_class_idx][0]
    else:
        class_shap = shap_values[0, :, pred_class_idx] if shap_values.ndim == 3 else shap_values[0]

    contributions = list(zip(feature_names, class_shap))
    # keep only features that are actually present/non-trivial for this post
    x_dense = X.toarray()[0]
    contributions = [(f, v) for f, v in contributions if x_dense[feature_names.index(f)] != 0]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "predicted_class": pred_class,
        "top_contributing_features": contributions[:top_k],
    }


def explain_with_lime(text, model, vectorizer, scaler, le, num_features=8):
    """LIME explanation: perturbs the input text and fits a local
    linear model to see which WORDS (not TF-IDF/linguistic features)
    push the prediction toward the predicted class."""

    def predict_proba(texts):
        X = featurize(texts, vectorizer, scaler)
        return model.predict_proba(X)

    explainer = LimeTextExplainer(class_names=list(le.classes_))
    exp = explainer.explain_instance(
        text, predict_proba, num_features=num_features,
        labels=list(range(len(le.classes_)))
    )
    pred_idx = np.argmax(predict_proba([text])[0])
    pred_class = le.inverse_transform([pred_idx])[0]
    word_contributions = exp.as_list(label=pred_idx)
    return {
        "predicted_class": pred_class,
        "top_words": word_contributions,
    }


def generate_explanation_report(text):
    model, vectorizer, scaler, le = load_artifacts()
    shap_result = explain_with_shap(text, model, vectorizer, scaler, le)
    lime_result = explain_with_lime(text, model, vectorizer, scaler, le)
    return {
        "input_text": text,
        "shap": shap_result,
        "lime": lime_result,
    }


if __name__ == "__main__":
    sample_posts = [
        "I don't even know why I'm posting this. Nothing feels worth doing anymore, I just lie in bed most of the day.",
        "My heart has been racing all day and I can't shake this feeling that something bad is about to happen.",
        "Work has been absolutely brutal this month, I'm juggling three deadlines and barely sleeping.",
        "Had a really nice weekend hiking with some friends, the weather was perfect for it.",
    ]
    for post in sample_posts:
        result = generate_explanation_report(post)
        print("\n" + "=" * 70)
        print("TEXT:", result["input_text"][:80], "...")
        print("Predicted class (SHAP):", result["shap"]["predicted_class"])
        print("Top SHAP features:")
        for feat, val in result["shap"]["top_contributing_features"]:
            print(f"    {feat:25s}  {val:+.4f}")
        print("Predicted class (LIME):", result["lime"]["predicted_class"])
        print("Top LIME words:")
        for word, val in result["lime"]["top_words"]:
            print(f"    {word:25s}  {val:+.4f}")