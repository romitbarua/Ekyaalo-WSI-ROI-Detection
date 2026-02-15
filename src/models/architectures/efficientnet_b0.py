"""
EfficientNet-B0 architecture for binary classification.

This matches the model architecture used during training in Ekyaalo_EfficientNet_B0_CNN.ipynb.
"""

import torch
import torch.nn as nn
try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required for EfficientNet-B0. Install it with: pip install timm"
    )


def create_efficientnet_b0(
    device: torch.device,
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """
    Create an EfficientNet-B0 model for binary classification.

    Caller must resolve and pass device (e.g. via get_device() or config).
    
    Args:
        device: Device to move the model to (required; caller owns device resolution)
        num_classes: Number of output classes (default: 2 for binary classification)
        pretrained: Whether to use pretrained weights (default: True)
    
    Returns:
        EfficientNet-B0 model ready for training/inference
    """
    # Create EfficientNet-B0 using timm
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    # Move model to device
    model = model.to(device)
    
    return model
