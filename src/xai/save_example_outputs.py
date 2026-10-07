"""Saves example SHAP/LIME explanations to results/ as JSON + a
bar-chart image, so they can be dropped straight into the review
slides as "Results Obtained So Far" evidence."""

import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.xai.explain import generate_explanation_report

RESULTS_DIR = "results"

EXAMPLES = [
    "I don't even know why I'm posting this. Nothing feels worth doing anymore, I just lie in bed most of the day.",
    "My heart has been racing all day and I can't shake this feeling that something bad is about to happen.",
]

all_results = []
for i, text in enumerate(EXAMPLES):
    result = generate_explanation_report(text)
    all_results.append(result)

    # Plot LIME word contributions (most intuitive for a non-technical panel)
    words, vals = zip(*result["lime"]["top_words"])
    colors = ["#d9534f" if v > 0 else "#5bc0de" for v in vals]
    plt.figure(figsize=(6, 4))
    plt.barh(words, vals, color=colors)
    plt.title(f"LIME explanation - predicted: {result['lime']['predicted_class']}")
    plt.xlabel("Contribution to predicted class")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fname = f"{RESULTS_DIR}/lime_example_{i+1}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

with open(f"{RESULTS_DIR}/xai_examples.json", "w") as f:
    # SHAP/LIME tuples aren't directly JSON serializable - convert
    serializable = []
    for r in all_results:
        serializable.append({
            "input_text": r["input_text"],
            "shap_predicted_class": r["shap"]["predicted_class"],
            "shap_top_features": [[f, float(v)] for f, v in r["shap"]["top_contributing_features"]],
            "lime_predicted_class": r["lime"]["predicted_class"],
            "lime_top_words": [[w, float(v)] for w, v in r["lime"]["top_words"]],
        })
    json.dump(serializable, f, indent=2)
print(f"Saved {RESULTS_DIR}/xai_examples.json")
