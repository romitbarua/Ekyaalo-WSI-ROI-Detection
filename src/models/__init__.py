"""
Model training and prediction utilities.
"""

from .predict_model import (
    load_model,
    create_resnet18_model,  # Deprecated, kept for backward compatibility
    load_resnet18_model,   # Deprecated, kept for backward compatibility
)
from .train_model import (
    TrainingConfig,
    train,
    setup_model,
    configure_training_mode,
)
from .architectures import (
    create_model,
    get_available_architectures,
    create_resnet18,
    create_efficientnet_b0,
    create_mobilenet_v3,
    create_baseline_cnn,
)

__all__ = [
    # Main functions
    'load_model',
    'create_model',
    'get_available_architectures',
    # Architecture-specific functions
    'create_resnet18',
    'create_efficientnet_b0',
    'create_mobilenet_v3',
    'create_baseline_cnn',
    # Training functions
    'TrainingConfig',
    'train',
    'setup_model',
    'configure_training_mode',
    # Deprecated (backward compatibility)
    'create_resnet18_model',
    'load_resnet18_model',
]

