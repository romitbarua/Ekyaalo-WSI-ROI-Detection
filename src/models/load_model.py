"""
Unified model loading from training configuration.

Handles both pretrained models (ResNet, EfficientNet, MobileNet) and
custom architectures (baseline_cnn, cytofm) with different parameter sets.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union

from src.utils.device import get_device
from .architectures import create_model
from .architectures.custom_cnn import create_baseline_cnn
from .architectures.cytofm import create_cytofm


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
        architecture: Model architecture name (e.g. resnet18, efficientnet_b0, cytofm)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (ignored for baseline_cnn and cytofm)
        model_params: Additional model parameters. For cytofm, checkpoint_path (absolute path
            to .pth file) is required. For baseline_cnn, optional architecture params.
        device: Device to load model on

    Returns:
        Initialized model ready for training/inference
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    arch_lower = architecture.lower().replace('-', '_')

    if arch_lower == 'cytofm':
        params = model_params or {}
        checkpoint_path = params.get('checkpoint_path')
        if not checkpoint_path or not str(checkpoint_path).strip():
            raise ValueError(
                "For architecture 'cytofm', model_params.checkpoint_path is required "
                "and must be a non-empty string (absolute path to the CytoFM checkpoint file)."
            )
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"CytoFM checkpoint not found: {checkpoint_path}. "
                "Please provide an absolute path to an existing .pth file."
            )
        return create_cytofm(
            device=device,
            num_classes=num_classes,
            checkpoint_path=str(path),
            **{k: v for k, v in params.items() if k != 'checkpoint_path'}
        )

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
