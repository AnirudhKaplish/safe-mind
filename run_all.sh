#!/bin/bash
# run_all.sh - runs the entire working pipeline end-to-end, in order.
# Run this from the project root: bash run_all.sh
set -e

echo "=================================================================="
echo "STEP 0: Build seed dataset"
echo "=================================================================="
python3 data/build_seed_dataset.py

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
echo "STEP 4: Generate sample assessment reports (allowed + blocked)"
echo "=================================================================="
python3 src/report/generate_report.py

echo ""
echo "=================================================================="
echo "STEP 5: Run the virtual hardware device demo"
echo "=================================================================="
python3 hardware/virtual_device.py

echo ""
echo "=================================================================="
echo "ALL STEPS COMPLETE. See results/ for metrics, plots, and reports."
echo "=================================================================="
