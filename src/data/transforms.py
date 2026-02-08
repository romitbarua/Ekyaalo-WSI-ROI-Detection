"""
Image transforms for WSI ROI Detection models.

This module provides consistent transforms for training and inference.
Transforms are designed to be saved with model checkpoints to ensure
consistency between training and inference.

Common patterns:
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Custom normalization: computed from dataset statistics
"""

from torchvision import transforms
from typing import Optional, Tuple, List, Dict, Any
import json
from pathlib import Path


# ImageNet normalization constants (standard for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



def get_train_transform(
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
    rotation: int = 15,
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    color_jitter: Optional[Dict[str, float]] = None,
    resize: Optional[Tuple[int, int]] = None
) -> transforms.Compose:
    """
    Get training transform with data augmentation.
    
    Args:
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
        rotation: Rotation angle in degrees (default: 15)
        horizontal_flip: Whether to apply horizontal flip (default: True)
        vertical_flip: Whether to apply vertical flip (default: True)
        color_jitter: Color jitter parameters dict with keys: brightness, contrast, saturation, hue
        resize: Optional resize tuple (height, width)
        
    Returns:
        Composed transform for training
    """
    transform_list = []
    
    if resize:
        transform_list.append(transforms.Resize(resize))
    
    # Data augmentation
    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    if vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip())
    
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))
    
    if color_jitter:
        transform_list.append(transforms.ColorJitter(**color_jitter))
    
    # Convert to tensor and normalize
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_val_transform(
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
    resize: Optional[Tuple[int, int]] = None
) -> transforms.Compose:
    """
    Get validation/inference transform (no augmentation).
    
    Args:
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
        resize: Optional resize tuple (height, width)
        
    Returns:
        Composed transform for validation/inference
    """
    transform_list = []
    
    if resize:
        transform_list.append(transforms.Resize(resize))
    
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def save_transform_config(
    transform_config: Dict[str, Any],
    save_path: Path
) -> None:
    """
    Save transform configuration to JSON file.
    
    Args:
        transform_config: Dictionary containing transform parameters
        save_path: Path to save the config file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(transform_config, f, indent=2)


def load_transform_config(load_path: Path) -> Dict[str, Any]:
    """
    Load transform configuration from JSON file.
    
    Args:
        load_path: Path to the config file
        
    Returns:
        Dictionary containing transform parameters
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Transform config not found: {load_path}")
    
    with open(load_path, 'r') as f:
        return json.load(f)


def create_transforms_from_config(config: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create train and validation transforms from a configuration dictionary.
    
    Args:
        config: Dictionary with transform parameters
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Extract normalization parameters
    mean = config.get('mean', IMAGENET_MEAN)
    std = config.get('std', IMAGENET_STD)
    
    # Extract augmentation parameters
    rotation = config.get('rotation', 15)
    horizontal_flip = config.get('horizontal_flip', True)
    vertical_flip = config.get('vertical_flip', True)
    color_jitter = config.get('color_jitter', None)
    resize = config.get('resize', None)
    
    # Ensure resize is set - default to [224, 224] if None to prevent batching errors
    if resize is None:
        resize = [224, 224]
        import warnings
        warnings.warn(
            "No resize specified in transform config. Using default [224, 224] "
            "to ensure consistent image sizes for batching. "
            "Set 'resize: [height, width]' in your config to avoid this warning."
        )
    
    # Convert list to tuple if needed (YAML loads lists as Python lists)
    if isinstance(resize, list):
        resize = tuple(resize)
    
    train_transform = get_train_transform(
        mean=mean,
        std=std,
        rotation=rotation,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        color_jitter=color_jitter,
        resize=resize
    )
    
    val_transform = get_val_transform(
        mean=mean,
        std=std,
        resize=resize
    )
    
    return train_transform, val_transform


# Default transform configuration (ImageNet normalization)
# Note: resize is set to [224, 224] by default to ensure all images are the same size for batching
DEFAULT_TRANSFORM_CONFIG = {
    'mean': IMAGENET_MEAN,
    'std': IMAGENET_STD,
    'rotation': 15,
    'horizontal_flip': True,
    'vertical_flip': True,
    'color_jitter': None,
    'resize': [224, 224]  # Default resize to 224x224 (standard ImageNet size)
}


class TransformFactory:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_train_transform(self) -> transforms.Compose:
        return get_train_transform(**self.config)

    def get_val_transform(self) -> transforms.Compose:
        return get_val_transform(**self.config)