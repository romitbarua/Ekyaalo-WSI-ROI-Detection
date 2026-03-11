#!/bin/bash
set -e
cd ~/wsi/Ekyaalo-WSI-ROI-Detection

echo "=== Step 1: Run k-fold for all CNN models (skip existing) ==="
.venv/bin/python cloud/run_kfold.py --folds 3 --no-shutdown --skip-existing

echo ""
echo "=== Step 2: Run k-fold for CytoFM frozen features + LogReg ==="
.venv/bin/python cloud/kfold_cytofm_frozen.py --folds 3

echo ""
echo "=== Step 3: Generate full report ==="
.venv/bin/python cloud/full_report.py 2>&1 | tee experiments/full_report_kfold.txt

echo ""
echo "=== DONE ==="

echo "Shutting down in 2 minutes..."
sudo shutdown -h +2
