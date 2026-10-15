#!/bin/bash
# run_all.sh - runs the entire working pipeline end-to-end, in order.
# Run this from the project root: bash run_all.sh
set -e

echo "=================================================================="
echo "STEP 0: Check the real dataset"
echo "=================================================================="
if [ ! -f data/seed_dataset.csv ]; then
  echo "data/seed_dataset.csv not found."
  echo "Download the Kaggle 'Depression: Reddit Dataset (Cleaned)' to data/real_dataset_raw.csv"
  echo "and run:  python3 data/prepare_real_dataset.py"
  echo "(data/build_seed_dataset.py only builds a small SYNTHETIC set - not used here.)"
  exit 1
fi
echo "Using data/seed_dataset.csv ($(($(wc -l < data/seed_dataset.csv) - 1)) rows)"

echo ""
echo "=================================================================="
echo "STEP 1: Train + evaluate baseline models (Logistic Regression, RF)"
echo "=================================================================="
python3 src/models/train_baselines.py

echo ""
echo "=================================================================="
echo "STEP 2: Generate SHAP/LIME explainability examples"
echo "=================================================================="
python3 src/xai/save_example_outputs.py

echo ""
echo "=================================================================="
echo "STEP 3: Deploy + demo the blockchain consent smart contract"
echo "=================================================================="
(cd blockchain && node scripts/compile.js && node scripts/deploy_and_demo.js)

echo ""
echo "=================================================================="
echo "STEP 4: Generate sample ensemble reports (baseline + BERT)"
echo "=================================================================="
if [ ! -d results/bert_model ]; then
  echo "results/bert_model/ not found - copy the fine-tuned BERT model there first"
  echo "(see src/models/finetune_transformer.py)."
  exit 1
fi
python3 src/report/generate_report_ensemble.py

echo ""
echo "=================================================================="
echo "STEP 5: Run the virtual hardware device demo (text; 3D twin: python3 -m robot_sim)"
echo "=================================================================="
python3 hardware/virtual_device.py

echo ""
echo "=================================================================="
echo "ALL STEPS COMPLETE. See results/ for metrics, plots, and reports."
echo "=================================================================="
