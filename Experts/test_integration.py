from pathlib import Path

import cv2
import numpy as np
import torch

from Base_CNN import SimpleCNNEncoder
from CellSAM_CNN import CellSAMCNN
from Shallow_DNN import ShallowDNN

# Paths

base_name = (
    "C1102-23-1.tif_lux_13817_luy_18392_h_216_w_156.png"
)

rgb_paths = [
    Path(
        r"C:\Users\ziaoo\Downloads"
    ) / base_name,

    Path(
        r"C:\Users\ziaoo\Downloads\cellsam_mvp_test\mild"
    ) / base_name,

    Path(
        r"C:\Users\ziaoo\Downloads\cellsam_mvp_test\moderate"
    ) / base_name,

    Path(
        r"C:\Users\ziaoo\Downloads\cellsam_mvp_test\severe"
    ) / base_name,
]


mask_directory = Path(
    r"C:\Users\ziaoo\Downloads\cellsam_mvp_masks"
)

mask_paths = [
    mask_directory / "clean_cellsam.npy",
    mask_directory / "mild_cellsam.npy",
    mask_directory / "moderate_cellsam.npy",
    mask_directory / "severe_cellsam.npy",
]

# Load RGB Images

rgb_tensors = []

for image_path in rgb_paths:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load RGB image: {image_path}"
        )

    # OpenCV loads images as BGR.
    # Convert them to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize to the size expected by the CNN.
    image = cv2.resize(
        image,
        (224, 224)
    )

    # Convert pixel values:
    # uint [0, 255] -> float32 [0, 1]
    image = image.astype(
        np.float32
    ) / 255.0

    # Convert: [height, width, channels] -> [channels, height, width]

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image_tensor = torch.from_numpy(
        image
    )

    rgb_tensors.append(
        image_tensor
    )

# Combine: clean, mild, moderate, severe into one batch: [4, 3, 224, 224]

rgb_images = torch.stack(
    rgb_tensors
)

# Load CellSAM masks

mask_tensors = []

for mask_path in mask_paths:

    if not mask_path.is_file():
        raise FileNotFoundError(
            f"Could not find CellSAM mask: {mask_path}"
        )

    mask = np.load(
        mask_path
    )

    # Resize the mask, INTER_NEAREST is important because CellSAM masks contain integer instance labels

    mask = cv2.resize(
        mask,
        (224, 224),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert NumPy uint32 mask into a PyTorch-supported integer representation

    mask_tensor = torch.from_numpy(
        mask
    ).long()

    # Convert: [height, width] -> [1, height, width]

    mask_tensor = mask_tensor.unsqueeze(
        0
    )

    mask_tensors.append(
        mask_tensor
    )

# Combine the four masks into: [4, 1, 224, 224]

cellsam_masks = torch.stack(
    mask_tensors
)

# Create MVP Model Components

rgb_cnn = SimpleCNNEncoder(
    input_channels=3,
    embedding_dim=128
)

cellsam_cnn = CellSAMCNN(
    embedding_dim=128
)

shallow_dnn = ShallowDNN(
    rgb_embedding_dim=128,
    cellsam_embedding_dim=128,
    hidden_dim=128,
    num_classes=2
)

# Forward Pass

rgb_embeddings = rgb_cnn(
    rgb_images
)

cellsam_embeddings = cellsam_cnn(
    cellsam_masks
)

# ShallowDNN concatenates the two embeddings and produces classification logits

output = shallow_dnn(
    rgb_embeddings,
    cellsam_embeddings
)

# Classification output
# Convert raw classification logits into probabilities

probabilities = torch.softmax(
    output,
    dim=1
)

# Select the class with the highest probability

predicted_classes = torch.argmax(
    probabilities,
    dim=1
)

# Backpropagation test

# Temporary labels used ONLY to test whether gradients can flow through the complete MVP. 
# All four inputs come from the same underlying image, so we assign the same temporary class here.
# These are NOT real training labels

labels = torch.tensor([
    0,
    0,
    0,
    0
])

# CrossEntropyLoss compares the model's logits against the target classes

criterion = torch.nn.CrossEntropyLoss()

loss = criterion(
    output,
    labels
)

# Send the classification error backward through: 
#
# classification layer
#       ↓
# shallow DNN
#       ↓
# concatenation
#      ↙   ↘
# RGB CNN  CellSAM CNN

loss.backward()

# Check gradient flow

rgb_has_gradients = (
    next(
        rgb_cnn.parameters()
    ).grad
    is not None
)

cellsam_has_gradients = (
    next(
        cellsam_cnn.parameters()
    ).grad
    is not None
)

dnn_has_gradients = (
    next(
        shallow_dnn.parameters()
    ).grad
    is not None
)

# Results

print(
    "CellSAM mask dtype:",
    cellsam_masks.dtype
)

print(
    "RGB input:",
    rgb_images.shape
)

print(
    "CellSAM input:",
    cellsam_masks.shape
)

print(
    "RGB embeddings:",
    rgb_embeddings.shape
)

print(
    "CellSAM embeddings:",
    cellsam_embeddings.shape
)

print(
    "Final output:",
    output.shape
)

print(
    "\nFinal logits:"
)

print(
    output
)

print(
    "\nProbabilities:"
)

print(
    probabilities
)

print(
    "\nPredicted classes:"
)

print(
    predicted_classes
)

print(
    "\nLoss:",
    loss.item()
)

print(
    "\nGradient flow:"
)

print(
    "RGB CNN gradients:",
    rgb_has_gradients
)

print(
    "CellSAM CNN gradients:",
    cellsam_has_gradients
)

print(
    "Shallow DNN gradients:",
    dnn_has_gradients
)

# Final validation

if (
    rgb_has_gradients
    and cellsam_has_gradients
    and dnn_has_gradients
):

    print(
        "\nMVP real-data forward and backward pass successful!"
    )

else:

    print(
        "\nWARNING: Gradient flow failed in part of the MVP."
    )