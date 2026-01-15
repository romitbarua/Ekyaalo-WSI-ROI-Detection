"""
ResNet-18 architecture for binary classification.

This matches the model architecture used during training in Ekyaalo_ResNet18_CNN.ipynb.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


def create_resnet18(
    num_classes: int = 2,
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Create a ResNet-18 model with a binary classification head.
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        device: Device to move the model to (default: auto-detect)
    
    Returns:
        ResNet-18 model with custom classification head
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
