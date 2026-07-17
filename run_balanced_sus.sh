#!/bin/bash
# 9 sus_balanced experiments — runs sequentially
cd /Users/romitbarua/BreastCancerResearch/Ekyaalo-WSI-ROI-Detection

python run_training.py --config configs/cnn_annotated_sus_balanced.yaml --metadata data/metadata/metadata_full_balanced.csv --experiment_name cnn_annotated_sus_balanced
python run_training.py --config configs/cnn_annotated_old_sus_balanced.yaml --metadata data/metadata/metadata_old_balanced.csv --experiment_name cnn_annotated_old_sus_balanced
python run_training.py --config configs/cnn_roi_sus_balanced.yaml --metadata data/metadata/metadata_roi_sus_balanced.csv --experiment_name cnn_roi_sus_balanced
python run_training.py --config configs/cytofm_head_annotated_sus_balanced.yaml --metadata data/metadata/metadata_full_balanced.csv --experiment_name cytofm_head_annotated_sus_balanced
python run_training.py --config configs/cytofm_head_annotated_old_sus_balanced.yaml --metadata data/metadata/metadata_old_balanced.csv --experiment_name cytofm_head_annotated_old_sus_balanced
python run_training.py --config configs/cytofm_head_roi_sus_balanced.yaml --metadata data/metadata/metadata_roi_sus_balanced.csv --experiment_name cytofm_head_roi_sus_balanced
python run_training.py --config configs/cytofm_finetune_annotated_sus_balanced.yaml --metadata data/metadata/metadata_full_balanced.csv --experiment_name cytofm_finetune_annotated_sus_balanced
python run_training.py --config configs/cytofm_finetune_annotated_old_sus_balanced.yaml --metadata data/metadata/metadata_old_balanced.csv --experiment_name cytofm_finetune_annotated_old_sus_balanced
python run_training.py --config configs/cytofm_finetune_roi_sus_balanced.yaml --metadata data/metadata/metadata_roi_sus_balanced.csv --experiment_name cytofm_finetune_roi_sus_balanced
