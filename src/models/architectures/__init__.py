"""
Model architecture definitions.

This module provides factory functions to create different model architectures
for binary classification on WSI ROI detection.
"""

from .resnet18 import create_resnet18
from .efficientnet_b0 import create_efficientnet_b0
from .mobilenet_v3 import create_mobilenet_v3
from .custom_cnn import create_baseline_cnn
from .factory import create_model, get_available_architectures

__all__ = [
    'create_resnet18',
    'create_efficientnet_b0',
    'create_mobilenet_v3',
    'create_baseline_cnn',
    'create_model',
    'get_available_architectures',
]
