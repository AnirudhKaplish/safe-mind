"""
dashboard/app.py
=================
Meridian - Doctor/Therapist Dashboard for the Mental-Health Assessment
system. Implements the "Doctor/Therapist Dashboard" component from
the approved design's Proposed Methodology diagram.

Fully self-contained: reuses src.xai.explain / src.models.bert_inference
/ src.features.input_type_filter directly (all proven to import
cleanly) rather than importing from src/report/*.py.

Run with:
    pip install flask
    python src/dashboard/app.py
Then open http://127.0.0.1:5000
"""

import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime, timezone
import numpy as np
from flask import Flask, request, redirect, url_for, render_template_string, jsonify
from lime.lime_text import LimeTextExplainer

from src.xai.explain import load_artifacts, featurize
try:
    from src.models.bert_inference import predict_proba as bert_predict_proba, load_bert
    BERT_AVAILABLE = os.path.isdir("results/bert_model")
except Exception:
    BERT_AVAILABLE = False
    bert_predict_proba = None
    load_bert = None
from src.features.input_type_filter import classify_input_type
from src.xai import highlight
from src.report import person_report
from src.preprocessing.clean_text import clean_pipeline
# Robot digital twin (robot_sim/): standard-library-only modules. The robot
# reads its state from /api/robot/state; it can never write back here.
from robot_sim.bus import robot_bus
from robot_sim.controller import from_report, processing_command, error_command

RESULTS_DIR = "results"
BASELINE_DEPRESSION_THRESHOLD = 0.40
BERT_DEPRESSION_THRESHOLD = 0.50
RISK_LEVEL_MAP = {"not_depression": "Low", "depression": "High"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # uploads up to 5 MB
REPORTS = []
PERSON_REPORTS = []


def get_consent_status(consent_log_path=f"{RESULTS_DIR}/blockchain_consent_demo_log.json"):
    if not os.path.exists(consent_log_path):
        return {"status": "unknown"}
    with open(consent_log_path) as f:
        log = json.load(f)
    return {
        "status": "revoked" if log.get("blockedConfirmed") else "granted",
        "contract_address": log.get("contractAddress"),
    }


def _baseline_proba_array(texts, vectorizer, scaler, model):
    X = featurize(texts, vectorizer, scaler)
    return model.predict_proba(X)


def explain_lime_baseline(text, model, vectorizer, scaler, le, pred_idx, num_features=6):
    """Model-agnostic LIME explanation for the baseline model - shows
    which words drove THIS specific prediction, not just the label."""
    explainer = LimeTextExplainer(class_names=list(le.classes_))

    def proba_fn(texts):
        return _baseline_proba_array(texts, vectorizer, scaler, model)

    exp = explainer.explain_instance(text, proba_fn, num_features=num_features, labels=[pred_idx])
    return exp.as_list(label=pred_idx)


def _bert_proba_array(texts):
    _, _, classes, _ = load_bert()
    dict_list = bert_predict_proba(list(texts))
    return np.array([[d[c] for c in classes] for d in dict_list])


def explain_lime_bert(text, pred_idx, num_features=6):
    """Same LIME approach applied to BERT. num_samples is reduced from
    LIME's default (5000) to 300 - BERT is much more expensive per
    forward pass than the baseline, and 300 perturbations still gives
    a stable explanation without an excessive wait."""
    _, _, classes, _ = load_bert()
    explainer = LimeTextExplainer(class_names=classes)
    exp = explainer.explain_instance(text, _bert_proba_array, num_features=num_features,
                                      labels=[pred_idx], num_samples=300)
    return exp.as_list(label=pred_idx)


def baseline_predict(text):
    model, vectorizer, scaler, le = load_artifacts()
    X = featurize([text], vectorizer, scaler)
    proba = model.predict_proba(X)[0]
    dep_idx = list(le.classes_).index("depression")
    pred_idx = dep_idx if proba[dep_idx] >= BASELINE_DEPRESSION_THRESHOLD else int(proba.argmax())
    pred_class = le.inverse_transform([pred_idx])[0]
    lime_words = explain_lime_baseline(text, model, vectorizer, scaler, le, pred_idx)
    return {"model": "Baseline", "model_full": "TF-IDF + Logistic Regression",
            "prediction": pred_class, "confidence": float(proba[pred_idx]),
            "dep_prob": float(proba[dep_idx]), "lime_words": lime_words}


def bert_predict(text):
    if BERT_AVAILABLE:
        proba_dict = bert_predict_proba([text])[0]
        dep_prob = proba_dict["depression"]
        _, _, classes, _ = load_bert()
        pred_idx = classes.index("depression") if dep_prob >= BERT_DEPRESSION_THRESHOLD else classes.index("not_depression")
        pred_class = classes[pred_idx]
        confidence = dep_prob if pred_class == "depression" else 1 - dep_prob
        lime_words = explain_lime_bert(text, pred_idx)
        return {"model": "BERT", "model_full": "Fine-tuned, calibrated transformer",
                "prediction": pred_class, "confidence": float(confidence),
                "dep_prob": float(dep_prob), "lime_words": lime_words}
    else:
        base = baseline_predict(text)
        return {"model": "BERT (Stand-in)", "model_full": "Baseline stand-in (results/bert_model not downloaded)",
                "prediction": base["prediction"], "confidence": base["confidence"],
                "dep_prob": base["dep_prob"], "lime_words": base["lime_words"]}


def build_report(text, patient_id, consent_log_path):
    consent = get_consent_status(consent_log_path)
    input_type = classify_input_type(text)

    if input_type["category"] != "personal_disclosure":
        return {
            "id": len(REPORTS) + 1, "patient_id": patient_id, "text": text,
            "generated_at": datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M"),
            "status": "NOT_APPLICABLE",
            "reason": input_type["reason"],
            "category": input_type["category"].replace("_", " "),
            "consent_status": consent,
        }

    if consent["status"] != "granted":
        return {
            "id": len(REPORTS) + 1, "patient_id": patient_id, "text": text,
            "generated_at": datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M"),
            "status": "BLOCKED",
            "reason": f"Consent status is '{consent['status']}'.",
            "consent_status": consent,
        }

    baseline_result = baseline_predict(text)
    bert_result = bert_predict(text)
    agree = baseline_result["prediction"] == bert_result["prediction"]

    report = {
        "id": len(REPORTS) + 1, "patient_id": patient_id, "text": text,
        "generated_at": datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M"),
        "status": "AGREEMENT" if agree else "DISAGREEMENT",
        "consent_status": consent,
        "model_breakdown": [baseline_result, bert_result],
    }
    report["explanation"] = highlight.explain(text, [baseline_result, bert_result])
    if agree:
        report["risk_class"] = baseline_result["prediction"]
        report["risk_level"] = RISK_LEVEL_MAP.get(baseline_result["prediction"], "Unknown")
        report["confidence"] = (baseline_result["confidence"] + bert_result["confidence"]) / 2
        avg_dep_prob = (baseline_result["dep_prob"] + bert_result["dep_prob"]) / 2
        report["meter_pct"] = round(avg_dep_prob * 100, 1)
    return report


# ---------------------------------------------------------------------------
# Batch helpers for person-level (many posts) reports
# ---------------------------------------------------------------------------
def baseline_dep_probs(texts):
    model, vectorizer, scaler, le = load_artifacts()
    proba = model.predict_proba(featurize(texts, vectorizer, scaler))
    dep_idx = list(le.classes_).index("depression")
    return [float(p[dep_idx]) for p in proba]


def bert_dep_probs(texts, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        out.extend(float(d["depression"]) for d in bert_predict_proba(list(texts[i:i + batch])))
    return out


def baseline_word_contribs(texts):
    """Exact per-word contribution of the linear baseline model towards
    'depression' for each text (coefficient x TF-IDF weight)."""
    model, vectorizer, scaler, le = load_artifacts()
    coef = model.coef_[0]
    dep_idx = list(le.classes_).index("depression")
    sign = 1.0 if list(model.classes_).index(dep_idx) == 1 else -1.0
    vocab = vectorizer.get_feature_names_out()
    tf = vectorizer.transform([clean_pipeline(t) for t in texts])
    out = []
    for r in range(tf.shape[0]):
        row = tf.getrow(r)
        out.append({vocab[j]: sign * coef[j] * v for j, v in zip(row.indices, row.data)})
    return out


def explain_post(text):
    return [baseline_predict(text), bert_predict(text)]


BASE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title|default('Meridian') }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#1C2A2E; --ink-soft:#28383D;
    --paper:#F6F5F1; --surface:#FFFFFF;
    --line:#E1E4DE; --line-soft:#ECEEE9;
    --text:#202B2A; --muted:#657573;
    --accent:#2F6F63; --accent-deep:#1F4C43;
    --risk-high:#B24B3C; --risk-low:#4C7A5D; --risk-review:#B08328; --risk-blocked:#8B958F;
    --radius:10px;
  }
  *{ box-sizing:border-box; }
  body{
    margin:0; background:var(--paper); color:var(--text);
    font-family:'Inter',sans-serif; font-size:15px; line-height:1.55;
    -webkit-font-smoothing:antialiased;
  }
  h1,h2,h3,.serif{ font-family:'Fraunces',serif; letter-spacing:-0.01em; }
  a{ color:var(--accent); }
  :focus-visible{ outline:2px solid var(--accent); outline-offset:2px; }

  .shell{ display:flex; min-height:100vh; }

  .sidebar{
    width:252px; flex-shrink:0; background:var(--ink); color:#EDF1EF;
    padding:28px 22px; position:relative; overflow:hidden;
  }
  .sidebar::before{
    content:""; position:absolute; inset:0; opacity:0.05; pointer-events:none;
    background-image: repeating-linear-gradient(115deg, transparent 0 26px, #ffffff 26px 27px);
  }
  .brand{ display:flex; align-items:center; gap:10px; position:relative; }
  .brand svg{ flex-shrink:0; }
  .brand-name{ font-family:'Fraunces',serif; font-size:20px; font-weight:500; }
  .brand-tag{ font-size:12px; color:#9FB3AE; margin:2px 0 0; }
  .nav{ margin-top:38px; position:relative; }
  .nav a{
    display:block; color:#C9D6D2; text-decoration:none; font-size:14px;
    padding:9px 12px; border-radius:8px; margin-bottom:2px; transition:background .15s;
  }
  .nav a:hover, .nav a.active{ background:rgba(255,255,255,0.08); color:#fff; }
  .sidebar-footer{ position:absolute; bottom:24px; left:22px; right:22px; }
  .consent-chip{
    display:flex; align-items:center; gap:8px; font-size:12.5px; color:#B9C8C3;
    border:1px solid rgba(255,255,255,0.14); border-radius:8px; padding:9px 11px;
  }
  .dot{ width:7px; height:7px; border-radius:50%; flex-shrink:0; }
  .dot.granted{ background:#6FBF9A; }
  .dot.revoked{ background:#D98066; }

  .main{ flex:1; padding:44px 48px; max-width:900px; }
  .eyebrow-free-title{ font-size:26px; margin:0 0 6px; font-weight:500; }
  .subtitle{ color:var(--muted); font-size:14.5px; margin:0 0 32px; max-width:56ch; }

  .panel{
    background:var(--surface); border:1px solid var(--line);
    border-radius:var(--radius); padding:26px 28px; margin-bottom:22px;
  }
  .panel h2{ font-size:16px; font-weight:500; margin:0 0 16px; color:var(--ink-soft); }

  label{ font-size:13px; color:var(--muted); display:block; margin-bottom:6px; }
  input[type=text], select{
    font-family:'Inter',sans-serif; font-size:14px; padding:9px 11px;
    border:1px solid var(--line); border-radius:7px; background:var(--paper);
    color:var(--text); width:100%;
  }
  textarea{
    font-family:'Inter',sans-serif; font-size:14.5px; width:100%; min-height:130px;
    padding:13px 14px; border:1px solid var(--line); border-radius:7px;
    background:var(--paper); color:var(--text); resize:vertical; line-height:1.6;
  }
  .field{ margin-bottom:16px; }
  .field-row{ display:flex; gap:16px; }
  .field-row .field{ flex:1; }

  button.primary{
    font-family:'Inter',sans-serif; font-weight:600; font-size:14px;
    background:var(--accent); color:#fff; border:none; padding:11px 22px;
    border-radius:7px; cursor:pointer; transition:background .15s, transform .1s;
  }
  button.primary:hover{ background:var(--accent-deep); }
  button.primary:active{ transform:scale(0.98); }

  .history{ position:relative; padding-left:6px; }
  .history-item{
    position:relative; padding:12px 0 12px 22px; border-bottom:1px solid var(--line-soft);
    font-size:14px;
  }
  .history-item::before{
    content:""; position:absolute; left:0; top:19px; width:8px; height:8px;
    border-radius:50%; background:var(--line);
  }
  .history-item.risk-high::before{ background:var(--risk-high); }
  .history-item.risk-low::before{ background:var(--risk-low); }
  .history-item.review::before{ background:var(--risk-review); }
  .history-item a{ color:var(--text); text-decoration:none; font-weight:500; }
  .history-item a:hover{ color:var(--accent); }
  .history-meta{ color:var(--muted); font-size:12.5px; margin-top:2px; }
  .empty{ color:var(--muted); font-size:14px; padding:8px 0; }

  .report-header{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:4px; }
  .report-meta{ color:var(--muted); font-size:13px; }
  .status-line{ display:flex; align-items:center; gap:8px; margin:18px 0 26px; }
  .status-pill{
    font-size:12.5px; font-weight:600; padding:4px 11px; border-radius:20px;
    border:1.5px solid currentColor;
  }
  .status-pill.high{ color:var(--risk-high); }
  .status-pill.low{ color:var(--risk-low); }
  .status-pill.review{ color:var(--risk-review); }
  .status-pill.blocked{ color:var(--risk-blocked); }

  .meter-wrap{ margin:18px 0 28px; }
  .meter-verdict{ font-size:32px; font-weight:500; margin-bottom:14px; }
  .meter-verdict.high{ color:var(--risk-high); }
  .meter-verdict.low{ color:var(--risk-low); }
  .meter-track{
    position:relative; height:10px; border-radius:6px; margin:0 6px;
    background:linear-gradient(90deg, var(--risk-low) 0%, #E4DCC4 50%, var(--risk-high) 100%);
  }
  .meter-marker{
    position:absolute; top:-7px; width:2px; height:24px; background:var(--ink);
    left:var(--target, 0%);
    animation: markerSlide 1s cubic-bezier(.2,.8,.2,1);
  }
  @keyframes markerSlide{ from{ left:0%; } to{ left:var(--target, 0%); } }
  .meter-marker::after{
    content:attr(data-pct); position:absolute; top:-24px; left:50%; transform:translateX(-50%);
    font-size:12px; font-weight:600; color:var(--ink); white-space:nowrap;
  }
  .meter-scale{ display:flex; justify-content:space-between; margin-top:8px; font-size:12px; color:var(--muted); }

  .split{ display:flex; border-radius:var(--radius); overflow:hidden; border:1px solid var(--line); margin:18px 0 26px; }
  .split-half{ flex:1; padding:20px 22px; position:relative; }
  .split-half.left{ background:#FBF4EE; }
  .split-half.right{ background:#F1F6F3; }
  .split-label{ font-size:12px; color:var(--muted); margin-bottom:4px; }
  .split-verdict{ font-family:'Fraunces',serif; font-size:19px; font-weight:500; }
  .split-conf{ font-size:13px; color:var(--muted); margin-top:2px; }

  .model-table{ width:100%; border-collapse:collapse; margin-top:6px; }
  .model-table td{ padding:9px 0; border-bottom:1px solid var(--line-soft); font-size:14px; }
  .model-table td.name{ color:var(--muted); }
  .model-table td.result{ text-align:right; font-weight:600; }

  .evidence-block{ margin-top:18px; }
  .evidence-label{ font-size:12px; color:var(--muted); margin-bottom:8px; font-weight:600; text-transform:uppercase; letter-spacing:0.03em; }
  .chip-row{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }
  .chip{
    display:inline-flex; align-items:center; gap:5px; padding:5px 12px; border-radius:20px;
    font-size:13px; border:1.5px solid; white-space:nowrap;
  }
  .chip.support-high{ color:var(--risk-high); border-color:var(--risk-high); background:#FBEEEC; }
  .chip.support-low{ color:var(--risk-low); border-color:var(--risk-low); background:#EEF4F0; }
  .chip.oppose{ color:var(--muted); border-color:var(--line); background:var(--paper); text-decoration:line-through; opacity:0.75; }
  .chip-val{ font-size:11px; opacity:0.8; }
  .evidence-note{ font-size:12.5px; color:var(--muted); margin-top:6px; }

  .quote-block{
    background:var(--paper); border-left:3px solid var(--line); border-radius:0 7px 7px 0;
    padding:14px 18px; font-size:14.5px; color:var(--ink-soft); line-height:1.65; margin-top:6px;
  }
  .note{ font-size:12.5px; color:var(--muted); margin-top:20px; padding-top:16px; border-top:1px solid var(--line-soft); }
  .back-link{ display:inline-block; margin-bottom:18px; font-size:13.5px; color:var(--muted); text-decoration:none; }
  .back-link:hover{ color:var(--accent); }

  #loading-overlay{
    position:fixed; inset:0; background:rgba(28,42,46,0.92); display:none;
    align-items:center; justify-content:center; flex-direction:column; z-index:50;
  }
  #loading-overlay.active{ display:flex; }
  .pulse-dots{ display:flex; gap:8px; margin-bottom:16px; }
  .pulse-dots span{
    width:10px; height:10px; border-radius:50%; background:#6FBF9A;
    animation:pulse 1.1s ease-in-out infinite;
  }
  .pulse-dots span:nth-child(2){ animation-delay:.15s; }
  .pulse-dots span:nth-child(3){ animation-delay:.3s; }
  @keyframes pulse{ 0%,80%,100%{ transform:scale(0.6); opacity:.5; } 40%{ transform:scale(1); opacity:1; } }
  .loading-text{ color:#D7E3DF; font-family:'Fraunces',serif; font-size:16px; }

  @media (prefers-reduced-motion: reduce){
    .meter-marker{ animation:none; }
    .pulse-dots span{ animation:none; opacity:1; }
  }

  /* plain-language result callouts */
  .callout{ border-radius:12px; padding:18px 20px; margin-bottom:18px; border:1.5px solid; }
  .callout h3{ margin:0 0 6px; font-family:'Fraunces',serif; font-size:21px; font-weight:600; }
  .callout p{ margin:6px 0 0; font-size:14.5px; line-height:1.5; }
  .callout.red{ background:#FDECEA; border-color:#C0392B; color:#6E1F17; }
  .callout.red h3{ color:#C0392B; }
  .callout.green{ background:#EAF5EE; border-color:#2E7D52; color:#1E4D34; }
  .callout.green h3{ color:#2E7D52; }
  .callout.amber{ background:#FFF5E1; border-color:#B08328; color:#5C4413; }
  .callout.amber h3{ color:#8A6418; }
  .callout.grey{ background:#F1F2F1; border-color:#8B958F; color:#3E4744; }
  .therapist{ display:block; color:#C0392B; font-weight:700; font-size:22px; margin:10px 0 2px; }
  .helpline{ margin-top:12px; padding-top:10px; border-top:1px dashed rgba(192,57,43,.4); font-size:13.5px; }
  .report-actions{ display:flex; gap:10px; align-items:flex-start; }
  .btn-pdf{ background:#2F6F63; color:#fff; border:0; border-radius:9px; padding:10px 16px;
            font:600 14px 'Inter',sans-serif; cursor:pointer; display:inline-flex; gap:8px; align-items:center; }
  .btn-pdf:hover{ background:#255A50; }
  .how-to-read{ font-size:13.5px; color:var(--ink-soft); line-height:1.55; }
  .how-to-read li{ margin:3px 0; }
  .print-only{ display:none; }
  .report-header{ display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }

  /* highlighted explanation */
  .why{ font-size:15px; line-height:1.55; color:var(--ink); margin:0 0 14px; }
  .hl-text{ font-size:16px; line-height:2.05; padding:16px 18px; background:#FBFAF7; border:1px solid var(--line-soft);
            border-radius:10px; white-space:pre-wrap; }
  .hl-text mark{ color:inherit; border-radius:4px; padding:1px 3px; }
  .hl-dep{ font-weight:600; }
  .hl-ok{ font-weight:500; }
  .hl-phrase{ border-bottom:2.5px solid #C0392B; padding-bottom:1px; }
  .hl-tag{ font-size:10px; font-weight:700; color:#fff; background:#C0392B; border-radius:4px; padding:1px 5px;
           margin-left:3px; letter-spacing:.2px; vertical-align:super; white-space:nowrap; }
  .hl-legend{ display:flex; flex-wrap:wrap; gap:8px 18px; font-size:12.5px; color:var(--muted); margin:10px 0 4px; }
  .hl-legend mark{ color:var(--ink); border-radius:4px; padding:0 4px; }
  .kw-table{ width:100%; border-collapse:collapse; margin-top:14px; font-size:13.5px; }
  .kw-table th{ text-align:left; font-size:11.5px; text-transform:uppercase; letter-spacing:.4px; color:var(--muted);
                font-weight:600; padding:6px 8px; border-bottom:1px solid var(--line); }
  .kw-table td{ padding:7px 8px; border-bottom:1px solid var(--line-soft); vertical-align:middle; }
  .kw-word{ font-weight:600; }
  .eff-up{ color:#C0392B; font-weight:600; } .eff-down{ color:#2E7D52; font-weight:600; } .eff-none{ color:#9AA39F; }
  .kw-bar{ height:8px; border-radius:4px; background:#EEE; position:relative; min-width:90px; }
  .kw-bar span{ position:absolute; left:0; top:0; bottom:0; border-radius:4px; }
  .theme-chip{ display:inline-block; background:#FDECEA; color:#8E2B20; border:1px solid #E8B4AD; border-radius:999px;
               padding:2px 10px; font-size:12.5px; font-weight:600; margin:2px 4px 2px 0; }
  .theme-chip.urgent{ background:#C0392B; color:#fff; border-color:#C0392B; }
  /* person report */
  .stats{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:6px 0 4px; }
  .stat{ background:#FBFAF7; border:1px solid var(--line-soft); border-radius:10px; padding:10px 12px; }
  .stat b{ display:block; font-size:24px; font-family:'Fraunces',serif; }
  .stat span{ font-size:12px; color:var(--muted); }
  .stat.red b{ color:#C0392B; } .stat.amber b{ color:#8A6418; } .stat.green b{ color:#2E7D52; }
  .ptable{ width:100%; border-collapse:collapse; font-size:13px; }
  .ptable th{ text-align:left; font-size:11px; text-transform:uppercase; color:var(--muted); padding:6px; border-bottom:1px solid var(--line); }
  .ptable td{ padding:7px 6px; border-bottom:1px solid var(--line-soft); vertical-align:top; }
  .dot{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
  .dot.flagged{ background:#C0392B; } .dot.unsure{ background:#B08328; } .dot.clear{ background:#2E7D52; }
  .post-card{ border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin:12px 0; break-inside:avoid; }
  .post-meta{ font-size:12.5px; color:var(--muted); margin-bottom:8px; }
  .muted-small{ font-size:12.5px; color:var(--muted); }
  .upload-hint{ font-size:13px; color:var(--ink-soft); background:#FBFAF7; border:1px dashed var(--line); border-radius:10px;
                padding:12px 14px; line-height:1.55; }
  .error-box{ background:#FDECEA; color:#8E2B20; border-radius:8px; padding:10px 14px; margin-bottom:12px; font-size:14px; }
  @media (max-width: 720px){ .stats{ grid-template-columns:repeat(2,1fr); } }

  @media print{
    @page{ margin:14mm; }
    *{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }
    body{ background:#fff; }
    .sidebar, .back-link, .report-actions, #loading-overlay{ display:none !important; }
    .shell{ display:block; }
    .main{ padding:0 !important; max-width:none !important; }
    .panel{ box-shadow:none; break-inside:auto; }
    .callout, .meter-wrap, .model-table, .evidence-block, .how-to-read, .quote-block{ break-inside:avoid; }
    .report-header{ display:block; }
    .eyebrow-free-title{ white-space:nowrap; font-size:26px !important; }
    .note{ break-before:avoid; }
    .print-only{ display:block; }
    .meter-marker{ animation:none; }
  }

  @media (max-width: 720px){
    .shell{ flex-direction:column; }
    .sidebar{ width:100%; }
    .main{ padding:28px 20px; }
  }
</style>
</head>
<body>
<div id="loading-overlay">
  <div class="pulse-dots"><span></span><span></span><span></span></div>
  <div class="loading-text">Running the assessment&hellip;</div>
  <div class="loading-text" style="font-size:13px;opacity:.75;margin-top:6px;">A person report with many posts can take a minute or two.</div>
</div>

<div class="shell">
  <div class="sidebar">
    <div class="brand">
      <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
        <rect width="30" height="30" rx="8" fill="#2F6F63"/>
        <path d="M5 16 Q9 9, 13 16 T21 16 T25 16" stroke="#EAF3F0" stroke-width="1.8" fill="none" stroke-linecap="round"/>
        <circle cx="25" cy="16" r="1.6" fill="#EAF3F0"/>
      </svg>
      <div>
        <div class="brand-name">Meridian</div>
        <p class="brand-tag">Assessment console</p>
      </div>
    </div>
    <div class="nav">
      <a href="{{ url_for('index') }}" class="{{ 'active' if active=='home' else '' }}">New assessment</a>
      <a href="{{ url_for('person_form') }}" class="{{ 'active' if active=='person' else '' }}">Person report (all posts)</a>
      <a href="{{ url_for('index') }}#history" class="{{ 'active' if active=='history' else '' }}">History</a>
    </div>
    <div class="sidebar-footer">
      <div class="consent-chip">
        <span class="dot {{ consent_status }}"></span>
        Consent {{ consent_status }}
      </div>
    </div>
  </div>
  <div class="main">
    {{ body|safe }}
  </div>
</div>

<script>
  document.addEventListener('DOMContentLoaded', function(){
    ['assess-form','person-form'].forEach(function(id){
      var form = document.getElementById(id);
      if(form){
        form.addEventListener('submit', function(){
          document.getElementById('loading-overlay').classList.add('active');
        });
      }
    });
  });
</script>
</body>
</html>
"""

EXPLAIN_MACRO = """
{% macro explain_block(e, title="Why the AI gave this result") %}
  <div class="panel">
    <h2>{{ title }}</h2>
    <p class="why">{{ e['why'] }}</p>
    <div class="hl-text">{{ e['html']|safe }}</div>
    <div class="hl-legend">
      <span><mark style="background:rgba(192,57,43,.55)">word</mark> pushed towards depression</span>
      <span><mark style="background:rgba(46,125,82,.45)">word</mark> pushed away from depression</span>
      <span><span class="hl-phrase">underlined</span> known warning phrase (theme in red)</span>
      <span>Darker colour = stronger effect</span>
    </div>
    {% if e['theme_names'] %}
      <p style="margin:12px 0 0; font-size:13.5px;"><strong>Warning themes found:</strong>
      {% for t in e['theme_names'] %}<span class="theme-chip {{ 'urgent' if t == 'Thoughts of disappearing' else '' }}">{{ t }}</span>{% endfor %}</p>
    {% endif %}
    {% if e['rows'] %}
    <table class="kw-table">
      <thead><tr><th>Word</th>
        {% for m in e['model_names'] %}<th>{{ 'Model 1 (word patterns)' if loop.index == 1 else 'Model 2 (BERT)' }}</th>{% endfor %}
        <th>Overall effect</th></tr></thead>
      <tbody>
      {% for row in e['rows'] %}
        <tr>
          <td class="kw-word">{{ row['word'] }}</td>
          {% for v in row['per_model'] %}
            <td>{% if v is none %}<span class="eff-none">not used</span>{% elif v > 0 %}<span class="eff-up">&uarr; towards depression</span>{% else %}<span class="eff-down">&darr; away from it</span>{% endif %}</td>
          {% endfor %}
          <td><div class="kw-bar"><span style="width:{{ row['bar'] }}%; background:{{ '#C0392B' if row['combined'] > 0 else '#2E7D52' }};"></span></div></td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    {% endif %}
  </div>
{% endmacro %}
"""

INDEX_BODY = """
<h1 class="eyebrow-free-title">New assessment</h1>
<p class="subtitle">Paste an authorized excerpt of a patient's text. Meridian checks consent, screens for whether it's a personal disclosure, then runs it through both models before showing you anything.</p>

<div class="panel">
  <form id="assess-form" method="POST" action="{{ url_for('analyze') }}">
    <div class="field-row">
      <div class="field">
        <label for="patient_id">Patient reference</label>
        <input type="text" id="patient_id" name="patient_id" value="demo-patient-001">
      </div>
      <div class="field">
        <label for="consent_scenario">Consent state to check against</label>
        <select id="consent_scenario" name="consent_scenario">
          <option value="granted">Currently granted</option>
          <option value="revoked">Currently revoked</option>
        </select>
      </div>
    </div>
    <div class="field">
      <label for="text">Patient text</label>
      <textarea id="text" name="text" placeholder="e.g. I haven't left my room in three days and I don't see the point in getting up anymore."></textarea>
    </div>
    <button type="submit" class="primary">Run assessment</button>
  </form>
</div>

<div class="panel" id="history">
  <h2>Recent assessments</h2>
  {% if reports %}
    <div class="history">
    {% for r in reports|reverse %}
      <div class="history-item {{ 'risk-high' if r.get('risk_level')=='High' else 'risk-low' if r.get('risk_level')=='Low' else 'review' if r['status']=='DISAGREEMENT' else '' }}">
        <a href="{{ url_for('view_report', report_id=r['id']) }}">{{ r['patient_id'] }}</a>
        <div class="history-meta">{{ r['generated_at'] }} &middot; {{ r['status'].replace('_',' ').title() }}</div>
      </div>
    {% endfor %}
    </div>
  {% else %}
    <p class="empty">Nothing yet. Run your first assessment above.</p>
  {% endif %}
</div>
"""

REPORT_BODY = EXPLAIN_MACRO + """
<a href="{{ url_for('index') }}" class="back-link">&lsaquo; Back to console</a>

<div class="print-only" style="margin-bottom:14px;">
  <div style="font-family:'Fraunces',serif; font-size:24px; font-weight:600; color:#1C2A2E;">Meridian &middot; Screening report</div>
  <div style="font-size:12.5px; color:#6B7672;">Report #{{ r['id'] }} &middot; generated {{ r['generated_at'] }} (UTC) &middot; for discussion with a doctor or therapist</div>
</div>
<div class="report-header">
  <div>
    <h1 class="eyebrow-free-title">{{ r['patient_id'] }}</h1>
    <p class="report-meta">{{ r['generated_at'] }}</p>
  </div>
  <div class="report-actions">
    <button type="button" class="btn-pdf" onclick="window.print()" title="Opens the print window. Choose 'Save as PDF'.">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 17v3h16v-3"/></svg>
      Save as PDF
    </button>
  </div>
</div>

{% if r['status'] == 'AGREEMENT' and r.get('risk_class') == 'depression' %}
  <div class="callout red">
    <h3>Signs of depression were found in this text</h3>
    <span class="therapist">Please meet a therapist.</span>
    <p>Both AI models agree. This is not a diagnosis, but it is a good reason to talk to a professional.
       Save this report as a PDF and take it with you: it shows the exact words that led to this result,
       so the therapist can see <em>why</em>.</p>
    <p class="helpline"><strong>Need to talk to someone now?</strong> Call Tele-MANAS on <strong>14416</strong>
       or <strong>1800-891-4416</strong> (free, 24/7, India).</p>
  </div>
{% elif r['status'] == 'AGREEMENT' %}
  <div class="callout green">
    <h3>No signs of depression were found</h3>
    <p>Both AI models agree that this text does not show signs of depression.
       If you are still feeling low, talking to someone you trust or a therapist always helps.</p>
  </div>
{% elif r['status'] == 'DISAGREEMENT' %}
  <div class="callout amber">
    <h3>The AI is not sure</h3>
    <p>The two AI models gave different answers, so no automatic result is given.
       Please show this report to a therapist so a person can decide.</p>
  </div>
{% elif r['status'] == 'BLOCKED' %}
  <div class="callout grey">
    <h3>Not analysed: no permission</h3>
    <p>Consent to analyse this text was withdrawn, so nothing was processed.</p>
  </div>
{% elif r['status'] == 'NOT_APPLICABLE' and r.get('category') == 'too short' %}
  <div class="callout grey">
    <h3>Too short to check</h3>
    <p>{{ r['reason'] }} Very short texts like "i am happy" can confuse the AI, so they are not screened.</p>
  </div>
{% elif r['status'] == 'NOT_APPLICABLE' %}
  <div class="callout grey">
    <h3>Not analysed: not a personal post</h3>
    <p>This text doesn't look like someone describing their own feelings, so it was not screened.</p>
  </div>
{% endif %}

{% if r['status'] == 'BLOCKED' %}
  <div class="status-line"><span class="status-pill blocked">Blocked</span></div>
  <div class="panel">
    <p>{{ r['reason'] }} Processing was not run.</p>
  </div>

{% elif r['status'] == 'NOT_APPLICABLE' %}
  <div class="status-line"><span class="status-pill blocked">{{ 'Too short' if r.get('category') == 'too short' else 'Not a personal disclosure' }}</span></div>
  <div class="panel">
    <p>Detected as <strong>{{ r['category'] }}</strong>. {{ r['reason'] }}</p>
    <p style="color:var(--muted); font-size:13.5px;">Risk classification does not apply to this kind of input, so it was not run.</p>
  </div>

{% elif r['status'] == 'DISAGREEMENT' %}
  <div class="status-line"><span class="status-pill review">Needs review</span></div>
  <div class="panel">
    <h2>The two models disagree</h2>
    <div class="split">
      <div class="split-half left">
        <div class="split-label">{{ r['model_breakdown'][0]['model_full'] }}</div>
        <div class="split-verdict">{{ r['model_breakdown'][0]['prediction'].replace('_',' ')|title }}</div>
        <div class="split-conf">{{ '%.0f' % (r['model_breakdown'][0]['confidence']*100) }}% confidence</div>
      </div>
      <div class="split-half right">
        <div class="split-label">{{ r['model_breakdown'][1]['model_full'] }}</div>
        <div class="split-verdict">{{ r['model_breakdown'][1]['prediction'].replace('_',' ')|title }}</div>
        <div class="split-conf">{{ '%.0f' % (r['model_breakdown'][1]['confidence']*100) }}% confidence</div>
      </div>
    </div>
    <p style="font-size:14px; color:var(--ink-soft);">Because the two models disagree, no automatic result is given. A therapist should review this text.</p>
  </div>
  {{ explain_block(r['explanation']) if r.get('explanation') }}

{% else %}
  <div class="status-line">
    <span class="status-pill {{ 'high' if r['risk_level']=='High' else 'low' }}">Risk: {{ r['risk_level'] }}</span>
  </div>
  <div class="panel">
    <div class="meter-wrap">
      <div class="meter-verdict {{ 'high' if r['risk_level']=='High' else 'low' }}">{{ r['risk_class'].replace('_',' ')|title }}</div>
      <div class="meter-track">
        <div class="meter-marker" style="--target: {{ r['meter_pct'] }}%;" data-pct="{{ '%.0f' % r['meter_pct'] }}%"></div>
      </div>
      <div class="meter-scale"><span>Not depression</span><span>Depression</span></div>
    </div>
    <h2>What each AI model said</h2>
    <table class="model-table">
      {% for m in r['model_breakdown'] %}
        <tr>
          <td class="name">{{ m['model_full'] }}</td>
          <td class="result">{{ m['prediction'].replace('_',' ')|title }} &middot; {{ '%.0f' % (m['confidence']*100) }}%</td>
        </tr>
      {% endfor %}
    </table>
  </div>
  {{ explain_block(r['explanation']) if r.get('explanation') }}
{% endif %}

{% if r['status'] in ('AGREEMENT', 'DISAGREEMENT') %}
<div class="panel">
  <h2>How to read this report</h2>
  <ul class="how-to-read">
    <li>Two different AI models read the text separately. When they agree, you get a result; when they disagree, a person should decide.</li>
    <li>The percentage shows how sure the models were, not how serious anything is.</li>
    <li>In the coloured text, red words pushed the AI towards depression and green words pushed it away. Underlined phrases are known warning phrases (for example about low energy or hopelessness). This lets a therapist check whether the AI understood the text correctly.</li>
    <li>This is a screening aid, not a diagnosis. Only a doctor or therapist can diagnose depression.</li>
  </ul>
</div>
{% endif %}

{% if not r.get('explanation') %}
<div class="panel">
  <h2>The text that was checked</h2>
  <div class="quote-block">{{ r['text'] }}</div>
</div>
{% endif %}

<p class="note">This is decision support, not an autonomous diagnosis. Final clinical judgement rests with the reviewing doctor or therapist.</p>
"""


PERSON_FORM_BODY = """
<h1 class="eyebrow-free-title">Person report</h1>
<p class="subtitle">Screen many posts by the same person at once, for example their Reddit history, and get one report:
an overall result, a timeline, recurring warning themes, and the posts that need attention.</p>

{% if error %}<div class="error-box">{{ error }}</div>{% endif %}
<div class="panel">
  <form id="person-form" method="POST" action="{{ url_for('analyze_person') }}" enctype="multipart/form-data">
    <div class="field-row">
      <div class="field">
        <label for="patient_id">Patient reference</label>
        <input type="text" id="patient_id" name="patient_id" value="demo-patient-001">
      </div>
      <div class="field">
        <label for="consent_scenario">Consent state to check against</label>
        <select id="consent_scenario" name="consent_scenario">
          <option value="granted">Currently granted</option>
          <option value="revoked">Currently revoked</option>
        </select>
      </div>
    </div>
    <div class="field">
      <label for="files">Upload the person's data (optional)</label>
      <input type="file" id="files" name="files" accept=".csv,.json,.txt" multiple>
    </div>
    <div class="upload-hint">
      <strong>How the person shares their Reddit posts (with consent):</strong>
      they request their own data at <em>reddit.com/settings/data-request</em>, download the zip Reddit emails them,
      and share <code>posts.csv</code> and/or <code>comments.csv</code>. You can upload both at once.
      JSON (<code>[{"text": ..., "date": ...}]</code>) and plain text files also work.
      Sample files are in <code>data\\samples\\</code>.
    </div>
    <div class="field" style="margin-top:14px;">
      <label for="posts">&hellip;or paste posts (leave a blank line between posts; optional date like <code>2026-05-01:</code> at the start)</label>
      <textarea id="posts" name="posts" style="min-height:160px;" placeholder="2026-05-01: Had a lovely weekend with my cousins, we went hiking and cooked dinner together.

2026-06-12: I haven't been sleeping well and I feel tired all the time lately."></textarea>
    </div>
    <button type="submit" class="primary">Build person report</button>
  </form>
</div>

<div class="panel">
  <h2>Previous person reports</h2>
  {% if reports %}
    <div class="history">
    {% for r in reports|reverse %}
      <div class="history-item">
        <a href="{{ url_for('view_person', report_id=r['id']) }}">{{ r['patient_id'] }}</a>
        <div class="history-meta">{{ r['generated_at'] }} &middot;
          {% if r['blocked'] %}Blocked (no consent){% else %}{{ r['s']['counts']['analysed'] }} posts checked &middot; {{ r['s']['verdict']['title'] }}{% endif %}</div>
      </div>
    {% endfor %}
    </div>
  {% else %}
    <p class="empty">Nothing yet.</p>
  {% endif %}
</div>
"""

PERSON_REPORT_BODY = EXPLAIN_MACRO + """
<a href="{{ url_for('person_form') }}" class="back-link">&lsaquo; Back to person reports</a>
<div class="print-only" style="margin-bottom:14px;">
  <div style="font-family:'Fraunces',serif; font-size:24px; font-weight:600; color:#1C2A2E;">Meridian &middot; Person screening report</div>
  <div style="font-size:12.5px; color:#6B7672;">Report P{{ r['id'] }} &middot; generated {{ r['generated_at'] }} (UTC) &middot; for discussion with a doctor or therapist</div>
</div>
<div class="report-header">
  <div>
    <h1 class="eyebrow-free-title">{{ r['patient_id'] }}</h1>
    <p class="report-meta">All posts &middot; {{ r['generated_at'] }}{% if not r['blocked'] and r['s']['period'] %} &middot; posts from {{ r['s']['period'][0] }} to {{ r['s']['period'][1] }}{% endif %}</p>
  </div>
  <div class="report-actions">
    <button type="button" class="btn-pdf" onclick="window.print()" title="Opens the print window. Choose 'Save as PDF'.">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v12m0 0l-4-4m4 4l4-4M4 17v3h16v-3"/></svg>
      Save as PDF
    </button>
  </div>
</div>

{% if r['blocked'] %}
  <div class="callout grey">
    <h3>Not analysed: no permission</h3>
    <p>Consent to analyse this person's posts was withdrawn, so none of the {{ r['n_posts'] }} posts were processed.</p>
  </div>
{% else %}
{% set s = r['s'] %}
{% set v = s['verdict'] %}

{% if s['urgent'] %}
  <div class="callout red" style="border-width:2.5px;">
    <h3>Some posts mention wanting to disappear</h3>
    <p><strong>Please reach out to the person soon.</strong> If there is any immediate danger, contact emergency services.
       Tele-MANAS (free, 24/7, India): <strong>14416</strong> or <strong>1800-891-4416</strong>.</p>
  </div>
{% endif %}

{% if v['level'] == 'red' %}
  <div class="callout red">
    <h3>{{ v['title'] }}</h3>
    <span class="therapist">Please meet a therapist.</span>
    <p>{{ s['counts']['flagged'] }} of {{ s['counts']['analysed'] }} checked posts were flagged by both AI models{% if s['trend'] == 'rising' %}, and the signs get stronger in the more recent posts{% endif %}.
       This is not a diagnosis. Save this report as a PDF and take it to a therapist: it shows which posts and which words led to this result.</p>
    <p class="helpline"><strong>Need to talk to someone now?</strong> Call Tele-MANAS on <strong>14416</strong> or <strong>1800-891-4416</strong> (free, 24/7, India).</p>
  </div>
{% elif v['level'] == 'amber' %}
  <div class="callout amber">
    <h3>{{ v['title'] }}</h3>
    <p>{{ s['counts']['flagged'] }} post(s) were flagged by both models and the models disagreed on {{ s['counts']['unsure'] }}.
       It is worth discussing these posts with a therapist.</p>
  </div>
{% elif v['level'] == 'green' %}
  <div class="callout green">
    <h3>{{ v['title'] }}</h3>
    <p>None of the {{ s['counts']['analysed'] }} checked posts were flagged by both AI models. If the person still feels low, talking to someone always helps.</p>
  </div>
{% else %}
  <div class="callout grey"><h3>{{ v['title'] }}</h3><p>{{ v['rule'] }}</p></div>
{% endif %}

<div class="panel">
  <h2>Summary</h2>
  <div class="stats">
    <div class="stat"><b>{{ s['counts']['analysed'] }}</b><span>posts checked</span></div>
    <div class="stat red"><b>{{ s['counts']['flagged'] }}</b><span>signs found (both AIs agree)</span></div>
    <div class="stat amber"><b>{{ s['counts']['unsure'] }}</b><span>not sure (AIs disagree)</span></div>
    <div class="stat green"><b>{{ s['counts']['clear'] }}</b><span>no signs</span></div>
    <div class="stat"><b>{{ s['counts']['skipped'] }}</b><span>skipped (too short / not personal)</span></div>
  </div>
  {% if s['subreddits'] %}<p class="muted-small" style="margin-top:10px;">Most posts were in:
    {% for name, n in s['subreddits'] %}r/{{ name }} ({{ n }}){{ ', ' if not loop.last }}{% endfor %}</p>{% endif %}
  <p class="muted-small" style="margin-top:6px;"><strong>How the overall result was decided:</strong> {{ v['rule'] }}</p>
</div>

{% if s['timeline'] %}
<div class="panel">
  <h2>Timeline</h2>
  <p class="muted-small" style="margin-top:-4px;">One dot per post, oldest on the left. Higher = the AI models leaned more towards depression.
    <span class="dot flagged"></span>signs found <span class="dot unsure"></span>not sure <span class="dot clear"></span>no signs
    {% if s['trend'] == 'rising' %}&middot; <strong style="color:#C0392B;">Trend: getting worse over time</strong>
    {% elif s['trend'] == 'falling' %}&middot; <strong style="color:#2E7D52;">Trend: improving over time</strong>
    {% elif s['trend'] == 'steady' %}&middot; Trend: steady{% endif %}</p>
  {{ r['svg']|safe }}
</div>
{% endif %}

{% if s['themes'] %}
<div class="panel">
  <h2>Warning themes that came up</h2>
  <table class="ptable">
    <thead><tr><th>Theme</th><th>In how many posts</th><th>Example phrases (as written)</th><th>Posts</th></tr></thead>
    <tbody>
    {% for t in s['themes'] %}
      <tr>
        <td><span class="theme-chip {{ 'urgent' if t['urgent'] else '' }}">{{ t['theme'] }}</span></td>
        <td>{{ t['posts'] }}</td>
        <td>{% for ph in t['phrases'] %}&ldquo;{{ ph }}&rdquo;{{ ', ' if not loop.last }}{% endfor %}</td>
        <td class="muted-small">#{{ t['post_numbers']|join(', #') }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
{% endif %}

{% if s['words'] %}
<div class="panel">
  <h2>Words that kept pushing towards depression</h2>
  <p class="muted-small" style="margin-top:-4px;">Across all checked posts, from Model 1 (word patterns). Longer bar = stronger overall push.</p>
  <table class="kw-table">
    <tbody>
    {% for w in s['words'] %}
      <tr><td class="kw-word" style="width:30%;">{{ w['word'] }}</td>
          <td class="muted-small" style="width:22%;">in {{ w['posts'] }} post{{ 's' if w['posts'] != 1 }}</td>
          <td><div class="kw-bar"><span style="width:{{ w['bar'] }}%; background:#C0392B;"></span></div></td></tr>
    {% endfor %}
    </tbody>
  </table>
</div>
{% endif %}

{% if s['attention'] %}
<div class="panel">
  <h2>Posts that need attention</h2>
  <p class="muted-small" style="margin-top:-4px;">The most concerning posts, explained word by word.</p>
  {% for p in s['attention'] %}
    <div class="post-card">
      <div class="post-meta"><span class="dot {{ p['status'] }}"></span><strong>Post #{{ p['n'] }}</strong>
        {% if p['date'] %}&middot; {{ p['date'].strftime('%d %b %Y') }}{% endif %}
        {% if p['subreddit'] %}&middot; r/{{ p['subreddit'] }}{% endif %}
        &middot; {{ {'flagged': 'Signs found (both AIs agree)', 'unsure': 'Not sure (AIs disagree)', 'clear': 'No signs'}[p['status']] }}
        &middot; Model 1: {{ '%.0f' % (p['base_prob']*100) }}% &middot; Model 2: {{ '%.0f' % (p['bert_prob']*100) }}% towards depression</div>
      <p class="why" style="font-size:14px;">{{ p['explanation']['why'] }}</p>
      <div class="hl-text">{{ p['explanation']['html']|safe }}</div>
    </div>
  {% endfor %}
  <div class="hl-legend">
    <span><mark style="background:rgba(192,57,43,.55)">word</mark> pushed towards depression</span>
    <span><mark style="background:rgba(46,125,82,.45)">word</mark> pushed away from depression</span>
    <span><span class="hl-phrase">underlined</span> known warning phrase</span>
  </div>
</div>
{% endif %}

<div class="panel">
  <h2>All posts</h2>
  <table class="ptable">
    <thead><tr><th>#</th><th>Date</th><th>Result</th><th>Towards depression</th><th>Themes</th><th>Post</th></tr></thead>
    <tbody>
    {% for p in s['posts'] %}
      <tr>
        <td>{{ p['n'] }}</td>
        <td class="muted-small" style="white-space:nowrap;">{{ p['date'].strftime('%d %b %Y') if p['date'] else '' }}</td>
        {% if p.get('skip_reason') %}
          <td class="muted-small">Skipped ({{ p['skip_reason'] }})</td><td></td><td></td>
        {% else %}
          <td style="white-space:nowrap;"><span class="dot {{ p['status'] }}"></span>{{ {'flagged': 'Signs found', 'unsure': 'Not sure', 'clear': 'No signs'}[p['status']] }}</td>
          <td>{{ '%.0f' % (p['avg_prob']*100) }}%</td>
          <td class="muted-small">{{ p['theme_names']|join(', ') }}</td>
        {% endif %}
        <td style="font-size:12.5px;">{{ p['text'][:160] }}{{ '&hellip;'|safe if p['text']|length > 160 }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

<div class="panel">
  <h2>How to read this report</h2>
  <ul class="how-to-read">
    <li>Each post was checked by two different AI models. A post counts as "signs found" only when both agree.</li>
    <li>Posts that are very short, are questions, or are about someone else are skipped, because the AI is not reliable on them.</li>
    <li>Warning themes come from a list of phrases linked to depression symptoms (for example low energy, hopelessness, withdrawal from people).</li>
    <li>In the coloured posts, red words pushed the AI towards depression and green words pushed it away.</li>
    <li>This is a screening aid, not a diagnosis. Only a doctor or therapist can diagnose depression.</li>
  </ul>
  {% if not r['bert_available'] %}<p class="muted-small">Note: the BERT model was not found, so Model 2 used a stand-in.</p>{% endif %}
</div>
{% endif %}

<p class="note">This is decision support, not an autonomous diagnosis. Final clinical judgement rests with the reviewing doctor or therapist.</p>
"""


@app.route("/")
def index():
    consent = get_consent_status()
    body = render_template_string(INDEX_BODY, reports=REPORTS)
    return render_template_string(BASE, body=body, active="home",
                                   consent_status=consent["status"], title="Meridian - New assessment")


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "").strip()
    patient_id = request.form.get("patient_id", "demo-patient-001").strip()
    scenario = request.form.get("consent_scenario", "granted")

    consent_log_path = (f"{RESULTS_DIR}/blockchain_consent_state_granted.json"
                         if scenario == "granted"
                         else f"{RESULTS_DIR}/blockchain_consent_demo_log.json")

    if not text:
        return redirect(url_for("index"))

    robot_bus.publish(processing_command())
    try:
        report = build_report(text, patient_id, consent_log_path)
    except Exception:
        robot_bus.publish(error_command(reason="analysis_exception"))
        raise
    REPORTS.append(report)
    robot_bus.publish(from_report(report, report_id=report.get("id")))
    return redirect(url_for("view_report", report_id=report["id"]))


@app.route("/api/robot/state")
def robot_state():
    """Read-only state for the robot digital twin (GET only). Carries the
    robot state and a sanitised report summary - never the text, the
    patient reference or explanation words."""
    resp = jsonify(robot_bus.snapshot())
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/report/<int:report_id>")
def view_report(report_id):
    r = next((rep for rep in REPORTS if rep["id"] == report_id), None)
    if r is None:
        return redirect(url_for("index"))
    consent = get_consent_status()
    body = render_template_string(REPORT_BODY, r=r)
    return render_template_string(BASE, body=body, active="home",
                                   consent_status=consent["status"], title=f"Meridian - {r['patient_id']}")


@app.route("/person")
def person_form():
    consent = get_consent_status()
    body = render_template_string(PERSON_FORM_BODY, reports=PERSON_REPORTS, error=request.args.get("error"))
    return render_template_string(BASE, body=body, active="person",
                                   consent_status=consent["status"], title="Meridian - Person report")


@app.route("/analyze_person", methods=["POST"])
def analyze_person():
    patient_id = request.form.get("patient_id", "demo-patient-001").strip() or "demo-patient-001"
    scenario = request.form.get("consent_scenario", "granted")
    consent_log_path = (f"{RESULTS_DIR}/blockchain_consent_state_granted.json"
                        if scenario == "granted" else f"{RESULTS_DIR}/blockchain_consent_demo_log.json")
    posts = []
    for f in request.files.getlist("files"):
        if f and f.filename:
            raw = f.read().decode("utf-8-sig", errors="replace")
            try:
                posts.extend(person_report.parse_any(raw, f.filename))
            except (ValueError, KeyError):
                return redirect(url_for("person_form", error=f"Could not read {f.filename}."))
    pasted = request.form.get("posts", "").strip()
    if pasted:
        posts.extend(person_report.parse_text(pasted))
    if not posts:
        return redirect(url_for("person_form", error="Add some posts first (paste them or upload a file)."))

    consent = get_consent_status(consent_log_path)
    rid = len(PERSON_REPORTS) + 1
    base = {"id": rid, "patient_id": patient_id, "consent_status": consent,
            "generated_at": datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M")}
    if consent["status"] != "granted":
        report = dict(base, blocked=True, n_posts=len(posts))
        PERSON_REPORTS.append(report)
        robot_bus.publish(from_report({"status": "BLOCKED"}, report_id=rid))
        return redirect(url_for("view_person", report_id=rid))

    robot_bus.publish(processing_command())
    try:
        summary = person_report.analyse(
            posts, baseline_dep_probs=baseline_dep_probs,
            bert_dep_probs=bert_dep_probs if BERT_AVAILABLE else None,
            explain_post=explain_post, baseline_word_contribs=baseline_word_contribs,
            thresholds=(BASELINE_DEPRESSION_THRESHOLD, BERT_DEPRESSION_THRESHOLD))
    except Exception:
        robot_bus.publish(error_command(reason="analysis_exception"))
        raise
    report = dict(base, blocked=False, s=summary, svg=person_report.timeline_svg(summary["timeline"]),
                  bert_available=BERT_AVAILABLE)
    PERSON_REPORTS.append(report)
    robot_bus.publish(from_report(person_report.robot_report(summary), report_id=rid))
    return redirect(url_for("view_person", report_id=rid))


@app.route("/person/<int:report_id>")
def view_person(report_id):
    r = next((rep for rep in PERSON_REPORTS if rep["id"] == report_id), None)
    if r is None:
        return redirect(url_for("person_form"))
    consent = get_consent_status()
    body = render_template_string(PERSON_REPORT_BODY, r=r)
    return render_template_string(BASE, body=body, active="person",
                                   consent_status=consent["status"], title=f"Meridian - {r['patient_id']} (all posts)")


# --- SAFE-MIND user app (connected user check-in page at /user) ---
from src.dashboard.user_api import register as register_user_app
register_user_app(app, baseline_dep_probs, bert_dep_probs if BERT_AVAILABLE else None, build_report, REPORTS,
                  robot_bus, processing_command, from_report, error_command,
                  (BASELINE_DEPRESSION_THRESHOLD, BERT_DEPRESSION_THRESHOLD))


if __name__ == "__main__":
    print("Starting Meridian dashboard at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)