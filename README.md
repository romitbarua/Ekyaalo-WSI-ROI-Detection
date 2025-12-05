# Ekyaalo WSI ROI Detection

This repository contains a set of Jupyter notebooks used to train, evaluate, and compare several deep learning architectures for Region of Interest (ROI) detection on Whole Slide Images. The work contributes to the Ekyaalo initiative focused on expanding access to cytology-based cancer diagnostics in low-resource environments.

The models evaluated represent a range of architectural families, from lightweight convolutional neural networks to transformer-based models. The objective is to benchmark differences in accuracy, computational efficiency, and robustness across these methods.

## Architectures Evaluated

- ResNet-18  
- EfficientNet-B0  
- MobileNet-V3  
- DINO Vision Transformer (ViT)

---

## Repository Contents

### Core Classification Workflows

**Ekyaalo_ResNet18_CNN.ipynb**  
End-to-end workflow for training, validating, and evaluating a ResNet-18 classifier.

**Ekyaalo_EfficientNet_B0_CNN.ipynb**  
Training and evaluation of EfficientNet-B0 under consistent preprocessing and k-fold structure.

**Ekyaalo_MobileNet_V3_CNN.ipynb**  
MobileNet-V3 implementation designed for lightweight and efficient ROI detection.

**Ekyaalo_Interactive_Heatmap.ipynb**  
Notebook for interactively visualizing model-generated heatmaps on slide tiles.

### Evaluation and Comparison

**Ekyaalo_Model_Comparison.ipynb**  
Consolidated comparison across all architectures, including k-fold summaries, performance plots, and metric aggregation.

**EfficientNet_B0_Advanced_Evaluation.ipynb**  
Extended diagnostics and evaluation for EfficientNet-B0.

**MobileNet_V3_Advanced_Evaluation.ipynb**  
Deeper analysis of MobileNet-V3 results.

**ResNet18_Advanced_Evaluation.ipynb**  
Additional fold-level metrics, ROC curves, and confusion matrices for ResNet-18.

---

## Purpose

ROI detection focuses computational effort on diagnostically informative regions of a large WSI. Instead of processing an entire slide at full resolution, tiles are extracted and classified to identify regions likely to contain significant cytological features.

This repository supports:

- Consistent tiling, normalization, and augmentation  
- Slide-level stratified k-fold validation  
- Training across multiple model families  
- Export of tile-level predictions and model weights  
- Generation of ROC curves, confusion matrices, and metric tables  
- Comparative study of CNN and transformer-based architectures  

---

## Installation

Clone the repository:


git clone https://github.com/Bpfrederick/Ekyaalo-WSI-ROI-Detection.git

cd Ekyaalo-WSI-ROI-Detection



Install dependencies:


pip install -r requirements.txt



Dataset paths are configured in the initial cells of each notebook and can be modified based on the user's environment.

---

## Running the Workflows

Each notebook is self-contained and can be executed in Jupyter Notebook, JupyterLab, or Google Colab.

Typical workflow:

1. Load and index WSI tiles  
2. Apply preprocessing and augmentations  
3. Construct stratified k-fold training and validation splits  
4. Train the selected architecture  
5. Save best model weights and tile-level predictions  
6. Generate ROC curves, confusion matrices, and summary metrics  
7. Use the comparison notebook to aggregate and compare performance across architectures  

Outputs generated include:

- `tile_preds_fold_{k}.csv`  
- Best model weights saved as `.pth` files  
- ROC and confusion matrix visualizations  
- Summary CSV files reporting accuracy, precision, recall, F1, sensitivity, specificity, and AUC  

---

## Notes

These notebooks are intended for research and exploratory model development. They are not validated for clinical use or diagnostic purposes. The repository is part of ongoing pathology and cytology research through the Ekyaalo program.

---

## License

This project is released under the MIT License.

---

## Contact

For questions or collaboration inquiries, please reach out through the GitHub profile associated with this repository.

