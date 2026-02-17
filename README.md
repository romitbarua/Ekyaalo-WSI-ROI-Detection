# Ekyaalo WSI ROI Detection

This repository will be used to train, evaluate, and compare several deep learning architectures for Region of Interest (ROI) detection on Whole Slide Images. The work contributes to the Ekyaalo initiative focused on expanding access to cytology-based cancer diagnostics in low-resource environments.

The models evaluated represent a range of architectural families, from lightweight convolutional neural networks to transformer-based models. The objective is to benchmark differences in accuracy, computational efficiency, and robustness across these methods.

## Architectures Evaluated

- ResNet-18  
- EfficientNet-B0  
- MobileNet-V3  
- DINO Vision Transformer (ViT)

---

## Repository Contents

### Data
Each image used in the dataset is a patch of a whole slide. The models ingest a metadata.csv file, load the images & apply various transformations for robustness.

The dataset is located in the `/data` directory. In the data directory, we have a metadata diretory, that houses the metadata.csv.

`
Metatadata.csv
     filepath: relative path to the iamge
     fname: name of the file
     class_name: classification of the image (malignant, suspicious, begnign)
     label: label to be predicted by the classifier
     slide_id: the id of the original whole slide
     width: width in pixels
     height: height in pixels
     split: designated the train/val/test split
`
In some cases the label and class_name may not match exact. For example, often times suspicious and malignant are combined together with a label of 1.

Each person using the repo will have to download the image data & save it in the data directory.

The dataset can be found here.
[dataset](https://drive.google.com/drive/u/1/folders/1OdmAmI53EVXWIM_z50ReOr-x3EP9GwzU)

If you don't have access, please reach out to Brendan Frederick

**Creating the metadata.csv**
If any changes are made to the dataset, including the adding or removal of images, we need to re-create the metadata.json.

To recreate the metadata, use /notebooks/data/generate_data_config.ipynb.
To use this notebook:
1. Update the USER_ROOT variable to your repo path
2. Update the label definition when calling `generate_metadata`
3. Update the `to_csv` path

### Prepping For Model Runs
When running a training, we need:
1. A metadata.csv
2. A run configuration

The goal of the configuration is to provide the model with all the key parameters to train a model. This includes architecture, training parameters, model parameters, data augmentations, etc.

An example configuration:
`
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
`

### Running Training
Training is kicked off from: `run_training.py`

Sample Command:
`python run_training.py --config configs/training_example.yaml --experiment_name my_experiment`

This will create a directory for this training run where we will store:
1. The model pt files
2. Plots & train/val metrics
3. Details about the training run

### Evaluation and Comparison

TBD

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

## Notes

These notebooks are intended for research and exploratory model development. They are not validated for clinical use or diagnostic purposes. The repository is part of ongoing pathology and cytology research through the Ekyaalo program.

---

## License

This project is released under the MIT License.

---

## Contact

For questions or collaboration inquiries, please reach out through the GitHub profile associated with this repository.

