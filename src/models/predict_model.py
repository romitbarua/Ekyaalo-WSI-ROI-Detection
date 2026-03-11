"""Prediction utilities for trained models."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union

from src.utils.device import get_device
from .architectures import create_model, get_available_architectures
from src.data.transforms import (
    get_val_transform,
    load_transform_config,
    create_transforms_from_config,
    DEFAULT_TRANSFORM_CONFIG
)


def load_model(
    checkpoint_path: Union[str, Path],
    architecture: str = 'resnet18',
    num_classes: int = 2,
    device: Optional[torch.device] = None,
    eval_mode: bool = True,
    transform_config_path: Optional[Union[str, Path]] = None
) -> tuple[nn.Module, torch.nn.Module]:
    """
    Load a trained model from a checkpoint file with its transform configuration.
    
    This function automatically creates the correct model architecture and loads
    the trained weights. Supports ResNet-18, EfficientNet-B0, and MobileNet-V3.
    Also loads the transform configuration to ensure consistent preprocessing.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        architecture: Name of the architecture (default: 'resnet18')
                    Options: 'resnet18', 'efficientnet_b0', 'mobilenet_v3'
        num_classes: Number of output classes (default: 2)
        device: Device to load the model on (default: auto-detect)
        eval_mode: Whether to set model to evaluation mode (default: True)
        transform_config_path: Path to transform config JSON file.
                              If None, tries to find it in the same directory as checkpoint.
                              If not found, uses default ImageNet transforms.
    
    Returns:
        Tuple of (model, transform) - Loaded model and transform for inference
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If architecture name is not supported
    
    Examples:
        >>> # Load ResNet-18 model with transforms
        >>> model, transform = load_model('path/to/resnet18.pth', architecture='resnet18')
        >>> 
        >>> # Load with explicit transform config path
        >>> model, transform = load_model(
        ...     'path/to/model.pth',
        ...     transform_config_path='path/to/transform_config.json'
        ... )
    """
    if device is None:
        device = get_device()
    
    # Create the model architecture
    model = create_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,  # We'll load trained weights, so don't need pretrained
        device=device
    )
    
    # Load the trained weights
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Set to evaluation mode
    if eval_mode:
        model.eval()
    
    # Load transform configuration. We use only transform_config.json (saved at training time);
    # we do not depend on the full training YAML.
    if transform_config_path is None:
        exp_dir = checkpoint_path.parent.parent
        transform_config_path = exp_dir / "configs" / "transform_config.json"
    
    transform_config_path = Path(transform_config_path)
    
    if transform_config_path.exists():
        transform_config = load_transform_config(transform_config_path)
        _, transform = create_transforms_from_config(transform_config)
    else:
        transform = get_val_transform()
    
    return model, transform


# Backward compatibility: keep old function names
def create_resnet18_model(num_classes: int = 2, device: Optional[torch.device] = None) -> nn.Module:
    """
    Create a ResNet-18 model with a binary classification head.

    Deprecated: Use create_model('resnet18', ...) instead.
    This function is kept for backward compatibility.
    """
    if device is None:
        device = get_device()
    return create_model('resnet18', num_classes=num_classes, device=device)


def load_resnet18_model(
    checkpoint_path: Union[str, Path],
    num_classes: int = 2,
    device: Optional[torch.device] = None,
    eval_mode: bool = True
) -> nn.Module:
    """
    Load a trained ResNet-18 model from a checkpoint file.
    
    Deprecated: Use load_model(..., architecture='resnet18') instead.
    This function is kept for backward compatibility.
    """
    return load_model(
        checkpoint_path=checkpoint_path,
        architecture='resnet18',
        num_classes=num_classes,
        device=device,
        eval_mode=eval_mode
    )


def main():
    """Main function for model prediction."""
    pass

if __name__ == "__main__":
    main()

