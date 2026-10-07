# Interpretable Mental Health Detection from Social Media

Implementation for the approved project design. This README covers what's
built, what's real vs. seeded, and exactly how to run it.

## What's implemented and WORKING (executable today)

| # | Module | Path | Status |
|---|--------|------|--------|
| 1 | Data pipeline (seed dataset) | `data/build_seed_dataset.py` | ✅ Working |
| 2 | Text preprocessing | `src/preprocessing/clean_text.py` | ✅ Working |
| 3 | Feature extraction (TF-IDF, sentiment, linguistic) | `src/features/extract_features.py` | ✅ Working |
| 4 | Baseline classification (LR, RF) | `src/models/train_baselines.py` | ✅ Working, real metrics |
| 5 | Explainable AI (SHAP + LIME) | `src/xai/explain.py` | ✅ Working, real outputs |
| 6 | Blockchain dynamic consent (real smart contract) | `blockchain/` | ✅ Working, real deployed contract |
| 7 | Report generation (with consent enforcement) | `src/report/generate_report.py` | ✅ Working |
| 8 | Hardware firmware (ESP32) | `hardware/mhd_hardware.ino` | ✅ Written, needs physical board to flash |
| 9 | Virtual hardware demo (no board needed) | `hardware/virtual_device.py` | ✅ Working |
| 10 | Transformer fine-tuning (BERT/RoBERTa) | `src/models/finetune_transformer.py` | ⏳ Script ready - run on your machine/Colab (needs internet to pull pretrained weights) |
| 11 | Doctor/therapist dashboard | - | Not yet built (next milestone) |

## Dataset in use: REAL DATA

This pipeline now runs on the **real Kaggle "Depression: Reddit Dataset
(Cleaned)"** (7,649 posts, binary depression/not_depression labels) —
`data/real_dataset_raw.csv` is the original download,
`data/seed_dataset.csv` is the converted `text`/`label` version the
pipeline actually reads (produced by `data/prepare_real_dataset.py`).

**Note on scope:** this dataset's labels are binary (depression /
not-depression) — it doesn't have separate anxiety/stress labels, so
classification was narrowed from the original 4-class design to match
what's genuinely annotated in the real data. See
`results/PANEL_REVIEW_SUMMARY.md` for the full reasoning and how to
present this to the panel.

If you want multi-class (anxiety/stress) results with real ground
truth later, the Dreaddit dataset (Turcan & McKeown, 2019) has a
binary stress label and could be merged in - ask and the conversion
script can be extended.

## How to run everything

```bash
pip install -r requirements.txt
cd blockchain && npm install && cd ..
bash run_all.sh
```

This runs, in order: dataset build → baseline training → SHAP/LIME
examples → blockchain consent demo (real deployed contract) → report
generation (both allowed + blocked consent scenarios) → virtual
hardware demo. All outputs land in `results/`.

### Running the individual pieces

```bash
# Baseline models + metrics/plots
python3 src/models/train_baselines.py

# Explainability
python3 src/xai/explain.py

# Blockchain consent (real local chain + real smart contract)
cd blockchain
node scripts/compile.js
node scripts/deploy_and_demo.js

# Structured report (with consent enforcement)
python3 src/report/generate_report.py

# Virtual hardware device (no physical ESP32 needed)
python3 hardware/virtual_device.py

# BERT/RoBERTa fine-tuning - RUN THIS ON YOUR OWN MACHINE/COLAB
cd src/models
python3 finetune_transformer.py --model bert-base-uncased
```

## Project structure

```
mhd_project/
├── data/
│   ├── build_seed_dataset.py   # generates seed_dataset.csv
│   └── seed_dataset.csv
├── src/
│   ├── preprocessing/clean_text.py
│   ├── features/extract_features.py
│   ├── models/
│   │   ├── train_baselines.py       # Logistic Regression + Random Forest
│   │   └── finetune_transformer.py  # BERT/RoBERTa (run locally/Colab)
│   ├── xai/
│   │   ├── explain.py               # SHAP + LIME
│   │   └── save_example_outputs.py
│   └── report/generate_report.py    # combines model + XAI + consent
├── blockchain/
│   ├── contracts/ConsentManager.sol # the actual smart contract
│   ├── scripts/compile.js           # solc-based compile (npm, no external binary)
│   └── scripts/deploy_and_demo.js   # deploys + runs full consent lifecycle
├── hardware/
│   ├── mhd_hardware.ino             # ESP32 firmware (Arduino)
│   └── virtual_device.py            # console simulator of the same logic
├── results/                          # all generated metrics/plots/reports
├── run_all.sh
└── requirements.txt
```

## Why the blockchain is a real smart contract, not a mock

`blockchain/contracts/ConsentManager.sol` is compiled with the `solc`
npm package (ships its own WASM compiler, no external binary download
needed) and deployed to a real local Ethereum-compatible chain via
`ganache`. `scripts/deploy_and_demo.js` runs the full lifecycle -
grant → verify → modify → verify → revoke → blocked attempt - with
real transaction hashes and a real revert on the final step. This is
genuine on-chain execution, not a simulated log.
