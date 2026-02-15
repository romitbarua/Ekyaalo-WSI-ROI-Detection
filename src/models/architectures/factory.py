"""
Factory function to create models by architecture name.

This provides a unified interface to create any supported architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Callable

from .resnet18 import create_resnet18
from .efficientnet_b0 import create_efficientnet_b0
from .mobilenet_v3 import create_mobilenet_v3
from .custom_cnn import create_baseline_cnn


# Registry of available architectures
ARCHITECTURE_REGISTRY: Dict[str, Callable] = {
    'resnet18': create_resnet18,
    'resnet-18': create_resnet18,
    'ResNet-18': create_resnet18,
    'efficientnet_b0': create_efficientnet_b0,
    'efficientnet-b0': create_efficientnet_b0,
    'EfficientNet-B0': create_efficientnet_b0,
    'mobilenet_v3': create_mobilenet_v3,
    'mobilenet-v3': create_mobilenet_v3,
    'MobileNet-V3': create_mobilenet_v3,
    'mobilenetv3': create_mobilenet_v3,
    'baseline_cnn': create_baseline_cnn,
    'baseline-cnn': create_baseline_cnn,
    'Baseline-CNN': create_baseline_cnn,
    'custom_cnn': create_baseline_cnn,  # Alias for backward compatibility
    'custom-cnn': create_baseline_cnn,
    'cnn': create_baseline_cnn,
}


def get_available_architectures() -> list:
    """
    Get a list of available architecture names.
    
    Returns:
        List of unique architecture names (normalized)
    """
    # Return unique normalized names
    unique_names = set()
    for name in ARCHITECTURE_REGISTRY.keys():
        # Normalize to lowercase with underscore
        normalized = name.lower().replace('-', '_')
        unique_names.add(normalized)
    return sorted(list(unique_names))


def create_model(
    architecture: str,
    device: torch.device,
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function to create a model by architecture name.

    Caller must resolve and pass device (e.g. via get_device() or config).
    
    Args:
        architecture: Name of the architecture (case-insensitive, supports various formats)
                     Options: 'resnet18', 'efficientnet_b0', 'mobilenet_v3', 'custom_cnn'
        device: Device to move the model to (required; caller owns device resolution)
        num_classes: Number of output classes (default: 2)
        pretrained: Whether to use pretrained weights (default: True)
                   Note: custom_cnn doesn't support pretrained weights
    
    Returns:
        Model instance ready for training/inference
    
    Raises:
        ValueError: If architecture name is not supported
    
    Examples:
        >>> device = get_device()
        >>> model = create_model('resnet18', num_classes=2, device=device)
    """
    # Normalize architecture name
    architecture = architecture.lower().replace('-', '_')
    
    if architecture not in ARCHITECTURE_REGISTRY:
        available = ', '.join(get_available_architectures())
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Available architectures: {available}"
        )
    
    # Get the creation function and call it
    create_fn = ARCHITECTURE_REGISTRY[architecture]
    
    # Baseline CNN doesn't support pretrained parameter
    if architecture in ['baseline_cnn', 'custom_cnn']:
        return create_fn(
            device,
            num_classes=num_classes,
        )
    else:
        return create_fn(
            device,
            num_classes=num_classes,
            pretrained=pretrained,
        )
