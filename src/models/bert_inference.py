"""
bert_inference.py
==================
Thin wrapper around the fine-tuned BERT model (trained on Colab,
downloaded, and placed at results/bert_model/) so the rest of the
project can call it the same simple way it calls the baseline model:
give it text, get back class probabilities.

Also applies TEMPERATURE SCALING if a calibrated temperature has been
saved (see calibrate_bert.py) - this corrects the model's raw
confidence scores, which we found to be overconfident (99%+ on wrong
answers) before calibration.
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "results/bert_model"
TEMPERATURE_PATH = "results/bert_temperature.json"

_model = None
_tokenizer = None
_classes = None
_temperature = 1.0  # 1.0 = no calibration applied (default until calibrate_bert.py is run)


def load_bert():
    """Loads the model/tokenizer once and caches them (loading BERT
    takes a couple seconds - don't repeat it per sentence)."""
    global _model, _tokenizer, _classes, _temperature

    if _model is not None:
        return _model, _tokenizer, _classes, _temperature

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"{MODEL_DIR} not found. Download the trained model from Colab "
            f"and place its files there (see project instructions)."
        )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.eval()

    # Class order: sklearn's LabelEncoder sorts alphabetically, and
    # that's what was used to build label_id during Colab training -
    # "depression" < "not_depression" alphabetically, so id 0 =
    # depression, id 1 = not_depression. This MUST match the encoder
    # used during training exactly, or predictions will be silently
    # swapped.
    _classes = ["depression", "not_depression"]

    if os.path.exists(TEMPERATURE_PATH):
        with open(TEMPERATURE_PATH) as f:
            _temperature = json.load(f)["temperature"]
        print(f"[bert_inference] Loaded calibration temperature: {_temperature:.3f}")
    else:
        print("[bert_inference] No calibration file found - using raw (uncalibrated) confidence scores. "
              "Run calibrate_bert.py to fix this.")

    return _model, _tokenizer, _classes, _temperature


def predict_proba(texts: list) -> "list[dict]":
    """Given a list of raw text strings, returns a list of
    {class_name: probability} dicts, with temperature scaling applied
    if available."""
    model, tokenizer, classes, temperature = load_bert()

    inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                        padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits

    calibrated_logits = logits / temperature
    probs = torch.softmax(calibrated_logits, dim=-1).numpy()

    return [dict(zip(classes, p)) for p in probs]


def predict_one(text: str) -> dict:
    """Convenience wrapper for a single sentence. Returns
    {'predicted_class':..., 'confidence':..., 'probabilities': {...}}"""
    proba_dict = predict_proba([text])[0]
    pred_class = max(proba_dict, key=proba_dict.get)
    return {
        "predicted_class": pred_class,
        "confidence": float(proba_dict[pred_class]),
        "probabilities": {k: float(v) for k, v in proba_dict.items()},
    }


if __name__ == "__main__":
    tests = [
        "I can't get out of bed, everything feels pointless and I just want to disappear.",
        "Had a great time at the party last night with my friends.",
    ]
    for t in tests:
        result = predict_one(t)
        print(f"\n{t[:60]}...")
        print(f"  -> {result['predicted_class']} ({result['confidence']*100:.1f}%)")
        print(f"     full probabilities: {result['probabilities']}")