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
import logging
from pathlib import Path
from .norm_constants import NORM_CONSTANTS

logger = logging.getLogger(__name__)


# Single source of truth for default transform options. Partial configs are merged with this
# in create_transforms_from_config; mean/std are string keys into NORM_CONSTANTS.
DEFAULT_TRANSFORM_CONFIG = {
    'mean': 'IMAGENET_MEAN',
    'std': 'IMAGENET_STD',
    'grayscale': False,
    'rotation': 15,
    'horizontal_flip': True,
    'vertical_flip': True,
    'color_jitter': None,
    'resize': [224, 224]
}

REQUIRED_TRANSFORM_KEYS = frozenset(DEFAULT_TRANSFORM_CONFIG.keys())


def resolve_transform_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge config with DEFAULT_TRANSFORM_CONFIG and validate required keys are present.
    Single place for merge logic; used by create_transforms_from_config and save_resolved_transform_config.
    """
    full = {**DEFAULT_TRANSFORM_CONFIG, **config}
    missing = REQUIRED_TRANSFORM_KEYS - set(full.keys())
    if missing:
        raise ValueError(f"Transform config missing required keys: {sorted(missing)}")
    return full


def log_transform_pipeline_to_dir(
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    log_dir: Path,
) -> None:
    """Log train/val pipelines to experiment_dir/logs/transform_pipeline.txt (and logger)."""
    def pipeline_str(t: transforms.Compose) -> str:
        return "\n".join(f"  {i + 1}. {repr(step)}" for i, step in enumerate(t.transforms))

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "transform_pipeline.txt"
    content = "=== Train ===\n" + pipeline_str(train_transform) + "\n\n=== Val ===\n" + pipeline_str(val_transform) + "\n"
    out_path.write_text(content)
    logger.info("Transform pipeline written to %s", out_path)


def get_train_transform(
    grayscale: bool = False,
    mean: List[float] = NORM_CONSTANTS["IMAGENET_MEAN"],
    std: List[float] = NORM_CONSTANTS["IMAGENET_STD"],
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

    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    
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
    grayscale: bool = False,
    mean: List[float] = NORM_CONSTANTS["IMAGENET_MEAN"],
    std: List[float] = NORM_CONSTANTS["IMAGENET_STD"],
    resize: Optional[Tuple[int, int]] = None
) -> transforms.Compose:
    """
    Get validation/inference transform (no augmentation).
    
    Args:
        grayscale: Whether to convert to single-channel (default: False)
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
        resize: Optional resize tuple (height, width)
        
    Returns:
        Composed transform for validation/inference
    """
    transform_list = []
    
    if resize:
        transform_list.append(transforms.Resize(resize))

    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    
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


def save_resolved_transform_config(config: Dict[str, Any], save_path: Path) -> None:
    """
    Resolve config (merge with defaults), validate, and save to JSON.
    Use this to persist the experiment's transform artifact so inference can load it.
    """
    resolved = resolve_transform_config(config)
    save_transform_config(resolved, save_path)


def create_transforms_from_config(config: Dict[str, Any], log_dir: Optional[Path] = None) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create train and validation transforms from a configuration dictionary.
    Config is merged with DEFAULT_TRANSFORM_CONFIG so partial configs (e.g. YAML overrides)
    are supported. All required keys must be present after merge; no silent defaults.
    
    Args:
        config: Dictionary with transform parameters (partial or full).
        log_dir: Optional directory to write transform pipeline log. If None, logging is skipped (e.g. for inference).
        
    Returns:
        Tuple of (train_transform, val_transform)
        
    Raises:
        ValueError: If any required key is missing after merge, or mean/std/grayscale are incompatible.
    """
    full = resolve_transform_config(config)

    mean_key = full['mean']
    std_key = full['std']
    grayscale = full['grayscale']
    rotation = full['rotation']
    horizontal_flip = full['horizontal_flip']
    vertical_flip = full['vertical_flip']
    color_jitter = full['color_jitter']
    resize = full['resize']

    mean = NORM_CONSTANTS[mean_key]
    std = NORM_CONSTANTS[std_key]

    # error check: ensure that they number of channls & the mean & std are compatible
    if grayscale:
        if len(mean) != 1 or len(std) != 1:
            raise ValueError("Grayscale mode requires a single mean and std value")
    else:
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("RGB mode requires three mean and std values")
    
    # Convert list to tuple if needed (YAML loads lists as Python lists)
    if isinstance(resize, list):
        resize = tuple(resize)
    
    train_transform = get_train_transform(
        grayscale=grayscale,
        mean=mean,
        std=std,
        rotation=rotation,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        color_jitter=color_jitter,
        resize=resize
    )
    
    val_transform = get_val_transform(
        grayscale=grayscale,
        mean=mean,
        std=std,
        resize=resize
    )

    if log_dir is not None:
        log_transform_pipeline_to_dir(train_transform, val_transform, log_dir)
    
    return train_transform, val_transform