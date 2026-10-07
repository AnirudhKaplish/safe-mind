"""
prepare_real_dataset.py
========================
Converts the raw Kaggle "Depression: Reddit Dataset (Cleaned)" CSV
into the `text` + `label` format the rest of this pipeline expects.

Kaggle source: kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
Raw columns  : clean_text, is_depression (0/1)
Output       : text, label ("not_depression" / "depression")

USAGE:
  1. Download the dataset (see the step-by-step guide) and place the
     raw CSV at: data/real_dataset_raw.csv
  2. Run this script:
       python data/prepare_real_dataset.py
  3. It writes data/seed_dataset.csv (overwriting the synthetic seed
     data) - the rest of the pipeline (train_baselines.py, explain.py,
     etc.) needs ZERO changes after this.

NOTE ON CLASSES: the Kaggle dataset is binary (depression / not
depression) - NOT the 4-class (normal/depression/anxiety/stress) setup
the seed dataset used. This is the HONEST choice: we report results on
the labels that actually exist in the real data, rather than inventing
anxiety/stress labels that were never annotated. If you later add the
Dreaddit dataset (which has a binary stress label) or a dataset with
genuine multi-class mental-health labels, this script can be extended -
ask and I'll adjust it.
"""

import pandas as pd
import sys
import os

RAW_PATH = "data/real_dataset_raw.csv"
OUT_PATH = "data/seed_dataset.csv"


def main():
    if not os.path.exists(RAW_PATH):
        print(f"ERROR: {RAW_PATH} not found.")
        print("Download the dataset first (see the step-by-step guide),")
        print(f"and save it at exactly this path: {RAW_PATH}")
        sys.exit(1)

    df = pd.read_csv(RAW_PATH)
    print("Raw columns found:", list(df.columns))

    # Kaggle's column names as of the current dataset version
    text_col = "clean_text" if "clean_text" in df.columns else df.columns[0]
    label_col = "is_depression" if "is_depression" in df.columns else df.columns[1]

    out = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": df[label_col].map({1: "depression", 0: "not_depression"}),
    })

    # Basic sanity cleaning: drop empty/near-empty posts
    out = out[out["text"].str.strip().str.len() > 5].reset_index(drop=True)
    out = out.drop_duplicates(subset="text").reset_index(drop=True)

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(out)} rows to {OUT_PATH}")
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
