"""
ResNet-18 architecture for binary classification.

This matches the model architecture used during training in Ekyaalo_ResNet18_CNN.ipynb.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_resnet18(
    device: torch.device,
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """
    Create a ResNet-18 model with a binary classification head.

    Caller must resolve and pass device (e.g. via get_device() or config).
    
    Args:
        device: Device to move the model to (required; caller owns device resolution)
        num_classes: Number of output classes (default: 2 for binary classification)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
    
    Returns:
        ResNet-18 model with custom classification head
    """
    # Create ResNet-18 with ImageNet pretrained weights
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    
    # Replace the final fully connected layer for binary classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Move model to device
    model = model.to(device)
    
    return model
