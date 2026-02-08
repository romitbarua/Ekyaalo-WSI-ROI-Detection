"""
Unified model loading from training configuration.

Handles both pretrained models (ResNet, EfficientNet, MobileNet) and
custom architectures (baseline_cnn) with different parameter sets.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union

from .architectures import create_model
from .architectures.custom_cnn import create_baseline_cnn


def load_model_from_config(
    architecture: str,
    num_classes: int = 2,
    pretrained: bool = True,
    model_params: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """
    Load model from training config parameters.
    
    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (ignored for baseline_cnn)
        model_params: Additional model parameters (only for baseline_cnn)
        device: Device to load model on
    
    Returns:
        Initialized model ready for training/inference
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    arch_lower = architecture.lower().replace('-', '_')
    
    if arch_lower in ['baseline_cnn', 'custom_cnn', 'cnn']:
        params = model_params or {}
        return create_baseline_cnn(
            num_classes=num_classes,
            device=device,
            **params
        )
    else:
        return create_model(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            device=device
        )
