"""
Scripts to make predictions using trained models.

This module contains functions for:
- Loading trained models
- Generating predictions on new data
- Exporting tile-level predictions
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union

from .architectures import create_model, get_available_architectures


def load_model(
    checkpoint_path: Union[str, Path],
    architecture: str = 'resnet18',
    num_classes: int = 2,
    device: Optional[torch.device] = None,
    eval_mode: bool = True
) -> nn.Module:
    """
    Load a trained model from a checkpoint file.
    
    This function automatically creates the correct model architecture and loads
    the trained weights. Supports ResNet-18, EfficientNet-B0, and MobileNet-V3.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        architecture: Name of the architecture (default: 'resnet18')
                    Options: 'resnet18', 'efficientnet_b0', 'mobilenet_v3'
        num_classes: Number of output classes (default: 2)
        device: Device to load the model on (default: auto-detect)
        eval_mode: Whether to set model to evaluation mode (default: True)
    
    Returns:
        Loaded model ready for inference
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If architecture name is not supported
    
    Examples:
        >>> # Load ResNet-18 model
        >>> model = load_model('path/to/resnet18.pth', architecture='resnet18')
        >>> 
        >>> # Load EfficientNet-B0 model
        >>> model = load_model('path/to/efficientnet.pth', architecture='efficientnet_b0')
        >>> 
        >>> # Load MobileNet-V3 model
        >>> model = load_model('path/to/mobilenet.pth', architecture='mobilenet_v3')
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    return model


# Backward compatibility: keep old function names
def create_resnet18_model(num_classes: int = 2, device: Optional[torch.device] = None) -> nn.Module:
    """
    Create a ResNet-18 model with a binary classification head.
    
    Deprecated: Use create_model('resnet18', ...) instead.
    This function is kept for backward compatibility.
    """
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

