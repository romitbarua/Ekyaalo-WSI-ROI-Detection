# Expert Models

# Overview:
The experts directory contains the neural network components used in the initial Ekyaalo classification architecture. 

The architecture combines feature representations extracted from RGB pathology images and their corresponding CellSAM segmentation masks. Each input is processed by a separate convolutional neural network (CNN), producing a fixed-dimensional feature embedding. 

These embeddings are concatenated and passed through a shallow deep neural network (DNN) to generate classification predictions. 

The initial implementation supports processing pathology images across different degradation conditions, including clean, mild, moderate, and severe. 

# Architecture:
The model consists of three primary components:
1. Base CNN: Extracts visual features from RGB pathology images.
2. CellSAM CNN: Extracts structural features from CellSAM segmentation masks.
3. Shallow DNN: Combines both feature representations and generates classification predictions. 

# File Descriptions:
## Base_CNN.py:
Implements SimpleCNNEncoder, a convolutional neural network designed to extract fixed-dimensional feature embeddings from RGB pathology images. 

The network consists of three convolutional layers and ReLU activations, two max-pooling operations, adaptive average pooling, and a final linear projection. 

### Architecture:
- Convolutional layer : 3 -> 32 channels
- ReLU activation
- Max pooling
- Convolutional layer: 32 -> 64 channels
- ReLU activation
- Max pooling
- Convolutional layer: 64 -> 128 channels
- ReLU activation
- Adaptive average pooling to 1 x 1 
- Linear projection to the target embedding dimension

### Input shape: [batch_size, 3, height, width]

### Output shape: [batch_size, 128]

The default embedding dimension is 128, although it can be modified during initialization. 

Adaptive average pooling allows the encoder to produce fixed-dimensional embeddings from different spatial input dimensions, provided the input is sufficiently large for the pooling operations. 

The encoder generates feature representations rather than classification predictions. 

## CellSAM_CNN.py:
Implements a CellSAMCNN, a convolutional neural network for extracting structural feature embeddings from CellSAM semgentation masks. 

The model reuses the SimpleCNNEncoder architecture with a single input channel.

CellSAM instance segmentation masks are converted into binary representations before being processed by the CNN.

### Binary mask representation:
- 0: Background
- 1: Detected cell region

This conversion prevents arbitrary CellSAM instance identification numbers from being interpreted as meaningful numerical values. 

### Input shape: [batch_size, 1, height, width]
### Output shape: [batch_size, 128]

The initial implementation extracts features from binary cellular occupancy masks. Individual cell identities are not preserved after binarization. 

## Shallow_DNN.py:
Implements ShallowDNN, a shallow neural network responsible for combining RGB and CellSAM feature embeddings and generating classification predictions. 

The model concatenates both embeddings along the feature dimension before passing the combined representation through two fully connected layers. 

### Default architecture:
- RGB embedding: 128 dimensions
- CellSAM embedding: 128 dimensions
- Combined representation: 256 dimensions
- Fully connected layer: 256 -> 128
- ReLU activation
- Output layer: 128 -> 2

### Input shapes:
- RGB embedding: [batch_size, 128]
- CellSAM embedding: [batch_size, 128]

### Output shape: [batch_size, num_classes]

The network produces raw classification logits. Softmax can be applied to convert these logits into class probabilities.

The number of output classes and hidden dimensions can be modified during initialization. 

## test_integration.py:
Provides an initial integration test for validating the forward and backward passes of the complete architecture. 

The script processes RGB pathology images and corresponding CellSAM segmentation masks across four degradation conditions:
1. Clean
2. Mild
3. Moderate
4. Severe

RGB images are resized to 224 x 224 pixels, converted to RGB format, normalized to [0, 1], and transformed into PyTorch tensors. 

CellSAM instance segmentation masks are resized using nearest-neighbor interpolation to preserve discrete instance labels. 

### The script performs the following operations:
1. Loads RGB pathology images and CellSAM segmentation masks
2. Initializes the Base CNN, CellSAM CNN, and Shallow DNN.
3. Extracts feature embeddings from both CNN branches
4. Concatenates the embeddings and generates classification logits
5. Applies softmax to obtain class probabilities
6. Computes cross-entropy loss using temporary class labels
7. Performs backpropagation through the complete architecture
8. Checks whether gradients are present in all three model components

### Expected tensor dimensions:

Component                                     Tensor Shape
RGB images                                    [4, 3, 224, 224]
CellSAM masks                                 [4, 1, 224, 224]
RGB embeddings                                [4, 128]
CellSAM embeddings                            [4, 128]
Combined embeddings                           [4, 256]
Classification logits                         [4, 2]

The combined embeddings are constructed internally by ShallowDNN. 

Important: This script is an integration test rather than a training or model-performance evaluation pipeline. 

The temporary class labels are used exclusively to verify that backpropagation executes successfully. 

Successful execution confirms that the model components can perform a forward pass and that gradients can propagate through the architecture. It does not establish classification accuracy or robustness. 

The current implementation uses local file paths that must be updated to match the execution environment. 

# Dependencies:
The model components and integration test require: 
- Python 
- PyTorch
- Torchvision
- NumPy
- OpenCV
- Pillow

Install the dependencies using:
    pip install torch torchvision numpy opencv-python pillow

# Current limitations:
The current implementation represents an initial minimum viable architecture. 
- Both CNN encoders are initialized without pretrained weights
- The integration test does not train the networks
- The architecture currently uses direct feature concatenation rather than attention-based fusion
- CellSAM instance masks are binarized, removing individual cell identities
- The integration test processes four degradation conditions derived from one underlying pathology image
- Classification accuracy and degradation robustness have not yet been quantitively evaluated
- The integration test checks gradient presence but does not independently verify numerical gradient correctness

# Future Development: 
Potential extensions include:
- Integration with the complete pathology image dataset
- Development of training and validation pipelines
- Quantitative evaluation across different degradation conditions
- Grad-CAM and integrated gradients for feature attribution
- Comparison of RGB and CellSAM feature representations
- Investigation of alternative feature fusion architectures
- Evaluation of model robustness to image degradation
- Development of more interpretable cellular feature representations

The current implementation establishes the initial dual-branch feature extraction and classification pipeline for subsequent experimentation. 