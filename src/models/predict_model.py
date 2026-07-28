"""Prediction utilities for trained models."""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import yaml

from src.utils.device import get_device
from .architectures import create_model, get_available_architectures
from .load_model import load_model_from_config
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

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    arch_norm = architecture.lower().replace("-", "_")

    # Special handling for CytoFM, which is constructed via load_model_from_config
    if arch_norm == "cytofm":
        # Infer training config path from experiment directory unless explicitly provided
        exp_dir = checkpoint_path.parent.parent
        training_config_path = exp_dir / "configs" / "training_config.yaml"
        if not training_config_path.exists():
            raise FileNotFoundError(
                f"CytoFM requires a training_config.yaml with model_params.checkpoint_path, "
                f"but none was found at {training_config_path}"
            )

        with training_config_path.open() as f:
            if training_config_path.suffix in (".yaml", ".yml"):
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)

        cfg_arch = cfg.get("architecture", architecture)
        cfg_num_classes = cfg.get("num_classes", num_classes)
        cfg_pretrained = cfg.get("pretrained", True)
        cfg_model_params = cfg.get("model_params", {})

        model = load_model_from_config(
            architecture=cfg_arch,
            num_classes=cfg_num_classes,
            pretrained=cfg_pretrained,
            model_params=cfg_model_params,
            device=device,
        )
    else:
        # Create the model architecture via the standard factory
        model = create_model(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=False,  # We'll load trained weights, so don't need pretrained
            device=device
        )
    
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
    """
    Command-line entry point for running inference with a trained model.

    Example usage:
        python -m src.models.predict_model \\
            --checkpoint experiments/my_exp/models/best_model.pth \\
            --input_dir path/to/images \\
            --output_csv predictions.csv \\
            --architecture resnet18 \\
            --num_classes 2
    """

    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet18",
        choices=get_available_architectures(),
        help="Model architecture used during training.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of output classes used during training.",
    )
    parser.add_argument(
        "--transform_config",
        type=str,
        default=None,
        help=(
            "Optional path to transform_config.json. "
            "If not provided, will look in the experiment directory."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory of input images for inference.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Single image path for inference.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to CSV file where predictions will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Optional device override (e.g. 'cpu', 'cuda', 'mps'). "
            "If not set, device is auto-detected."
        ),
    )

    args = parser.parse_args()

    if not args.input_dir and not args.image_path:
        raise SystemExit("You must provide either --input_dir or --image_path.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve device using the same helper as training/inference utilities
    device = get_device(args.device)

    # Load model and its associated transform
    model, transform = load_model(
        checkpoint_path=checkpoint_path,
        architecture=args.architecture,
        num_classes=args.num_classes,
        device=device,
        eval_mode=True,
        transform_config_path=args.transform_config,
    )

    class ImageDataset(Dataset):
        """Simple dataset for loading images from disk."""

        def __init__(self, image_paths: List[Path], transform):
            self.image_paths = image_paths
            self.transform = transform

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
            path = self.image_paths[idx]
            img = Image.open(path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, path.name

    # Collect image paths
    image_paths: List[Path] = []
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise SystemExit(f"Input directory not found: {input_dir}")
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
        for pattern in patterns:
            image_paths.extend(sorted(input_dir.glob(pattern)))
    if args.image_path:
        p = Path(args.image_path)
        if not p.exists():
            raise SystemExit(f"Image not found: {p}")
        image_paths.append(p)

    if not image_paths:
        raise SystemExit("No images found for inference.")

    dataset = ImageDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for images, filenames in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for fname, pred, prob_vec in zip(filenames, preds_np, probs_np):
                row = {
                    "filename": fname,
                    "pred_label": int(pred),
                }
                for i, p in enumerate(prob_vec):
                    row[f"prob_{i}"] = float(p)
                rows.append(row)

    fieldnames = ["filename", "pred_label"]
    if rows:
        extra_keys = sorted(k for k in rows[0].keys() if k not in fieldnames)
        fieldnames.extend(extra_keys)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions for {len(rows)} images to {output_path}")

if __name__ == "__main__":
    main()

