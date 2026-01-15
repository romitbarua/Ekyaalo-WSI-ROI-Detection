"""
EfficientNet-B0 architecture for binary classification.

This matches the model architecture used during training in Ekyaalo_EfficientNet_B0_CNN.ipynb.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required for EfficientNet-B0. Install it with: pip install timm"
    )


def create_efficientnet_b0(
    num_classes: int = 2,
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Create an EfficientNet-B0 model for binary classification.
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        pretrained: Whether to use pretrained weights (default: True)
        device: Device to move the model to (default: auto-detect)
    
    Returns:
        EfficientNet-B0 model ready for training/inference
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create EfficientNet-B0 using timm
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    # Move model to device
    model = model.to(device)
    
    return model
