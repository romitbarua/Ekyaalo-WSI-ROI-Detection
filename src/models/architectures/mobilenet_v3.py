"""
MobileNet-V3 architecture for binary classification.

This matches the model architecture used during training in Ekyaalo_MobileNet_V3_CNN.ipynb.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required for MobileNet-V3. Install it with: pip install timm"
    )


def create_mobilenet_v3(
    num_classes: int = 2,
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Create a MobileNet-V3-Small model for binary classification.
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        pretrained: Whether to use pretrained weights (default: True)
        device: Device to move the model to (default: auto-detect)
    
    Returns:
        MobileNet-V3-Small model ready for training/inference
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create MobileNet-V3-Small using timm
    model = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    # Move model to device
    model = model.to(device)
    
    return model
