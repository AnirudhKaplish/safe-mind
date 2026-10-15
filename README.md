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
| 7 | Report generation (baseline + BERT ensemble, consent enforcement) | `src/report/generate_report_ensemble.py` | ✅ Working (needs `results/bert_model/`) |
| 8 | Hardware firmware (ESP32) | `hardware/mhd_hardware.ino` | ✅ Written, needs physical board to flash |
| 9 | Virtual hardware demo, text (no board needed) | `hardware/virtual_device.py` | ✅ Working |
| 10 | Transformer fine-tuning (BERT/RoBERTa) | `src/models/finetune_transformer.py` | ✅ Trained on Colab; **required** at `results/bert_model/` by the dashboard and ensemble report (not in git) |
| 11 | Doctor/therapist dashboard (Meridian) | `src/dashboard/app.py` | ✅ Working (baseline + BERT + LIME, consent-gated) |
| 12 | Robot digital twin of the ESP32 companion | `robot_sim/` | ✅ Working, driven live by the dashboard (see below) |

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
python3 src/report/generate_report_ensemble.py

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
│   ├── dashboard/app.py             # Meridian dashboard (Flask)
│   └── report/generate_report_ensemble.py  # baseline + BERT + consent
├── blockchain/
│   ├── contracts/ConsentManager.sol # the actual smart contract
│   ├── scripts/compile.js           # solc-based compile (npm, no external binary)
│   └── scripts/deploy_and_demo.js   # deploys + runs full consent lifecycle
├── hardware/
│   ├── mhd_hardware.ino             # ESP32 firmware (Arduino)
│   └── virtual_device.py            # console simulator of the same logic
├── robot_sim/                        # 3D digital twin of the companion (PyBullet)
├── tests/robot_sim/                  # robot + dashboard-integration tests
├── docs/ROBOT_SIMULATION.md          # twin design, maths, payload contract
├── results/                          # all generated metrics/plots/reports
├── run_all.sh
└── requirements.txt
```

**Consent at analysis time is log-based:** the dashboard and reports read
the consent state the contract demo wrote (`results/blockchain_consent_state_granted.json`,
or `results/blockchain_consent_demo_log.json`, where `blockedConfirmed` means
revoked). They do not query the chain live during an assessment.

## Robot digital twin (ESP32 companion)

`robot_sim/` is a software twin of the ESP32 device in `hardware/`: a
pan–tilt head on two SG90 servos, a 128×64 OLED face, an RGB LED, a
buzzer and two buttons, simulated in PyBullet and shown in an OpenCV
console. The dashboard publishes each assessment's outcome at
`GET /api/robot/state` (read-only); the twin polls it and moves to the
matching state. It shows the *system's* result (analysing, screen clear,
indicators flagged, models disagree, consent blocked, not a personal
post, error, offline) — never emotions, clinical words, the input text,
the patient reference or explanation words.

```bash
pip install -r requirements.txt -r requirements-robot.txt
./run_robot_demo.sh                  # dashboard + robot (Windows: run_robot_demo.bat)
python -m robot_sim --manual         # demo all states without the models
python -m pytest tests/robot_sim -q  # tests (no BERT weights needed)
python -m robot_sim.selftest         # PASS/WARN/FAIL self-test
```

Full design, maths and the payload contract: `docs/ROBOT_SIMULATION.md`;
review walkthrough: `docs/robot_demo_script.md`.

## Why the blockchain is a real smart contract, not a mock

`blockchain/contracts/ConsentManager.sol` is compiled with the `solc`
npm package (ships its own WASM compiler, no external binary download
needed) and deployed to a real local Ethereum-compatible chain via
`ganache`. `scripts/deploy_and_demo.js` runs the full lifecycle -
grant → verify → modify → verify → revoke → blocked attempt - with
real transaction hashes and a real revert on the final step. This is
genuine on-chain execution, not a simulated log.
