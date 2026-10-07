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
from flask import Flask, request, redirect, url_for, render_template_string
from lime.lime_text import LimeTextExplainer

from src.xai.explain import load_artifacts, featurize
from src.models.bert_inference import predict_proba as bert_predict_proba, load_bert
from src.features.input_type_filter import classify_input_type

RESULTS_DIR = "results"
BASELINE_DEPRESSION_THRESHOLD = 0.40
BERT_DEPRESSION_THRESHOLD = 0.50
RISK_LEVEL_MAP = {"not_depression": "Low", "depression": "High"}

app = Flask(__name__)
REPORTS = []


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
    if agree:
        report["risk_class"] = baseline_result["prediction"]
        report["risk_level"] = RISK_LEVEL_MAP.get(baseline_result["prediction"], "Unknown")
        report["confidence"] = (baseline_result["confidence"] + bert_result["confidence"]) / 2
        avg_dep_prob = (baseline_result["dep_prob"] + bert_result["dep_prob"]) / 2
        report["meter_pct"] = round(avg_dep_prob * 100, 1)
    return report


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
    var form = document.getElementById('assess-form');
    if(form){
      form.addEventListener('submit', function(){
        document.getElementById('loading-overlay').classList.add('active');
      });
    }
  });
</script>
</body>
</html>
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

REPORT_BODY = """
<a href="{{ url_for('index') }}" class="back-link">&lsaquo; Back to console</a>

<div class="report-header">
  <div>
    <h1 class="eyebrow-free-title">{{ r['patient_id'] }}</h1>
    <p class="report-meta">{{ r['generated_at'] }}</p>
  </div>
</div>

{% if r['status'] == 'BLOCKED' %}
  <div class="status-line"><span class="status-pill blocked">Blocked</span></div>
  <div class="panel">
    <p>{{ r['reason'] }} Processing was not run.</p>
  </div>

{% elif r['status'] == 'NOT_APPLICABLE' %}
  <div class="status-line"><span class="status-pill blocked">Not a personal disclosure</span></div>
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
        <div class="chip-row" style="margin-top:12px;">
          {% for word, val in r['model_breakdown'][0]['lime_words'] %}
            {% set pred0 = r['model_breakdown'][0]['prediction'] %}
            {% if val > 0 and pred0 == 'depression' %}
              <span class="chip support-high">{{ word }}</span>
            {% elif val > 0 and pred0 == 'not_depression' %}
              <span class="chip support-low">{{ word }}</span>
            {% else %}
              <span class="chip oppose">{{ word }}</span>
            {% endif %}
          {% endfor %}
        </div>
      </div>
      <div class="split-half right">
        <div class="split-label">{{ r['model_breakdown'][1]['model_full'] }}</div>
        <div class="split-verdict">{{ r['model_breakdown'][1]['prediction'].replace('_',' ')|title }}</div>
        <div class="split-conf">{{ '%.0f' % (r['model_breakdown'][1]['confidence']*100) }}% confidence</div>
        <div class="chip-row" style="margin-top:12px;">
          {% for word, val in r['model_breakdown'][1]['lime_words'] %}
            {% set pred1 = r['model_breakdown'][1]['prediction'] %}
            {% if val > 0 and pred1 == 'depression' %}
              <span class="chip support-high">{{ word }}</span>
            {% elif val > 0 and pred1 == 'not_depression' %}
              <span class="chip support-low">{{ word }}</span>
            {% else %}
              <span class="chip oppose">{{ word }}</span>
            {% endif %}
          {% endfor %}
        </div>
      </div>
    </div>
    <p style="font-size:14px; color:var(--ink-soft);">No automatic verdict is given when the models disagree. Route this one to a clinician directly.</p>
  </div>

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
    <h2>Model breakdown</h2>
    <table class="model-table">
      {% for m in r['model_breakdown'] %}
        <tr>
          <td class="name">{{ m['model_full'] }}</td>
          <td class="result">{{ m['prediction'].replace('_',' ')|title }} &middot; {{ '%.0f' % (m['confidence']*100) }}%</td>
        </tr>
      {% endfor %}
    </table>

    <div class="evidence-block">
      <div class="evidence-label">Why: {{ r['model_breakdown'][0]['model'] }}</div>
      <div class="chip-row">
        {% for word, val in r['model_breakdown'][0]['lime_words'] %}
          {% if val > 0 and r['risk_class'] == 'depression' %}
            <span class="chip support-high">{{ word }}</span>
          {% elif val > 0 and r['risk_class'] == 'not_depression' %}
            <span class="chip support-low">{{ word }}</span>
          {% else %}
            <span class="chip oppose">{{ word }}</span>
          {% endif %}
        {% endfor %}
      </div>
      <div class="evidence-label">Why: {{ r['model_breakdown'][1]['model'] }}</div>
      <div class="chip-row">
        {% for word, val in r['model_breakdown'][1]['lime_words'] %}
          {% if val > 0 and r['risk_class'] == 'depression' %}
            <span class="chip support-high">{{ word }}</span>
          {% elif val > 0 and r['risk_class'] == 'not_depression' %}
            <span class="chip support-low">{{ word }}</span>
          {% else %}
            <span class="chip oppose">{{ word }}</span>
          {% endif %}
        {% endfor %}
      </div>
      <p class="evidence-note">Solid-colored words pushed toward the final verdict; crossed-out words argued against it.</p>
    </div>
  </div>
{% endif %}

<div class="panel">
  <h2>Patient text</h2>
  <div class="quote-block">{{ r['text'] }}</div>
</div>

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

    report = build_report(text, patient_id, consent_log_path)
    REPORTS.append(report)
    return redirect(url_for("view_report", report_id=report["id"]))


@app.route("/report/<int:report_id>")
def view_report(report_id):
    r = next((rep for rep in REPORTS if rep["id"] == report_id), None)
    if r is None:
        return redirect(url_for("index"))
    consent = get_consent_status()
    body = render_template_string(REPORT_BODY, r=r)
    return render_template_string(BASE, body=body, active="home",
                                   consent_status=consent["status"], title=f"Meridian - {r['patient_id']}")


if __name__ == "__main__":
    print("Starting Meridian dashboard at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)