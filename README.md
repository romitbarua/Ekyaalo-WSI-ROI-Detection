# Ekyaalo WSI ROI Detection

This repository is for training, evaluating, and comparing deep learning architectures for Region of Interest (ROI) detection on Whole Slide Images (WSIs). The work contributes to the Ekyaalo initiative, focused on expanding access to cytology-based cancer diagnostics in low-resource environments.

The models evaluated represent a range of architectural families, from lightweight convolutional neural networks to transformer-based models. The objective is to benchmark differences in accuracy, computational efficiency, and robustness across these methods.

## Architectures Evaluated

- ResNet-18  
- EfficientNet-B0  
- MobileNet-V3  
- DINO Vision Transformer (ViT)

---

## Repository Contents

### Data

Each image in the dataset is a patch of a whole slide. The models ingest a `metadata.csv` file, load the images, and apply various transformations for robustness.

The dataset is located in the `/data` directory. In this directory, you will find a `metadata` subdirectory that houses the `metadata.csv` file.

```
metadata.csv columns:
    filepath:    relative path to the image
    fname:       name of the file
    class_name:  classification of the image (malignant, suspicious, benign)
    label:       label to be predicted by the classifier
    slide_id:    ID of the original whole slide
    width:       width in pixels
    height:      height in pixels
    split:       designated train/val/test split
```

In some cases the `label` and `class_name` may not match exactly. For example, suspicious and malignant are often combined with a label of 1.

**Dataset Download:**  
[Dataset Download Link](https://drive.google.com/drive/u/1/folders/1OdmAmI53EVXWIM_z50ReOr-x3EP9GwzU)

If you don't have access, please reach out to Brendan Frederick.

When saving down the dataset, save down the AnnotatedDataSet from the google drive in data/raw

**Creating the `metadata.csv`:**  
If any changes are made to the dataset (adding or removing images), you will need to re-create the `metadata.csv`.

To recreate it, use `/notebooks/data/generate_data_config.ipynb`.  
Steps:
1. Update the `USER_ROOT` variable to your repo path
2. Update the label definition when calling `generate_metadata`
3. Update the path in `to_csv` for output

---

### Prepping for Model Runs

To run training, you need:  
1. A `metadata.csv`  
2. A run configuration YAML file

The configuration file provides the model with all key parameters for training, including architecture, training parameters, model parameters, data augmentations, etc.

**Example configuration:**

```yaml
architecture: "baseline_cnn"
num_classes: 2
pretrained: false

transforms:
  rotation: 15
  horizontal_flip: true
  vertical_flip: true
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
  resize: [224, 224]  # Resize all images to 224x224 for consistent batching
  mean: "RGB_MEAN_224"
  std: "RGB_STD_224"
  grayscale: false

model_params:
  base_channels: 32
  num_blocks: 4
  dropout: 0.5
  input_channels: 3

training_mode: "fine_tune"

batch_size: 32
num_epochs: 50
learning_rate: 0.001

optimizer: "adam"
optimizer_params:
  weight_decay: 0.0001

loss_function: "cross_entropy"

random_seed: 42
```

---

### Running Training

Training is kicked off from: `run_training.py`

**Sample Command:**
```sh
python run_training.py --config configs/training_example.yaml --experiment_name my_experiment
```

This will create a directory for this training run where we will store:
1. The model `.pt` files
2. Plots & train/val metrics
3. Details about the training run

---

### Evaluation and Comparison

_TBD_

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

```sh
git clone https://github.com/Bpfrederick/Ekyaalo-WSI-ROI-Detection.git
cd Ekyaalo-WSI-ROI-Detection
```

Install dependencies:

```sh
pip install -r requirements.txt
```

Dataset paths are configured in the initial cells of each notebook and can be modified based on your environment.

---

## Notes

These notebooks are intended for research and exploratory model development. They are not validated for clinical use or diagnostic purposes. The repository is part of ongoing pathology and cytology research through the Ekyaalo program.

---

## License

This project is released under the MIT License.

---

## Contact

For questions or collaboration inquiries, please reach out through the GitHub profile associated with this repository.

