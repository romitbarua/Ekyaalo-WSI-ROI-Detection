#!/bin/bash
cd /Users/romitbarua/BreastCancerResearch/Ekyaalo-WSI-ROI-Detection

python run_training.py --config configs/cnn_roi_square_nosus_balanced.yaml --metadata data/metadata/metadata_roi_square_nosus_balanced.csv --experiment_name cnn_roi_square_nosus_balanced
python run_training.py --config configs/cytofm_head_roi_square_nosus_balanced.yaml --metadata data/metadata/metadata_roi_square_nosus_balanced.csv --experiment_name cytofm_head_roi_square_nosus_balanced
python run_training.py --config configs/cytofm_finetune_roi_square_nosus_balanced.yaml --metadata data/metadata/metadata_roi_square_nosus_balanced.csv --experiment_name cytofm_finetune_roi_square_nosus_balanced
