#!/usr/bin/env python3
"""
Main training script for WSI ROI Detection models.

This script:
- Creates experiment directories
- Loads and splits data
- Sets up data loaders
- Runs training with configurable parameters
- Saves models, metrics, and configs to experiment directory

Usage:
    python run_training.py --config configs/training_example.yaml
    python run_training.py --config configs/training_example.yaml --experiment_name my_experiment
"""

import argparse
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import json
import yaml
import shutil
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split

from src.models.train_model import train, TrainingConfig
from src.utils.logging_config import setup_logging
from src.utils.validate_config import validate_training_paths
from src.data.slide_dataloader import create_data_loader
from src.data.transforms import (
    DEFAULT_TRANSFORM_CONFIG,
    create_transforms_from_config,
    save_resolved_transform_config,
)

logger = logging.getLogger(__name__)


def create_experiment_dir(base_dir: str = "experiments", experiment_name: str = None) -> Path:
    """
    Create a new experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional custom name for experiment
        
    Returns:
        Path to the created experiment directory
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    if experiment_name:
        exp_dir = base_path / (experiment_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    else:
        exp_dir = base_path / datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "configs").mkdir(exist_ok=True)
    (exp_dir / "metrics").mkdir(exist_ok=True)
    
    return exp_dir


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load metadata CSV file."""
    return pd.read_csv(metadata_path)


def prepare_data_loaders(
    metadata: pd.DataFrame,
    train_config: TrainingConfig,
    transform_config: Dict[str, Any],
    val_split: float = 0.2,
    test_split: float = 0.1,
    random_seed: int = 42,
    log_dir: Path = None,
    no_aug_train_loader: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Prepare train, validation, and test data loaders.
    
    Args:
        metadata: Full metadata dataframe
        config: Training configuration
        val_split: Fraction of data for validation
        test_split: Fraction of data for test
        random_seed: Random seed for reproducibility
        no_aug_train_loader: Whether to create a train-set eval loader without augmentation to measure impact of augmentation
    Returns:
        Tuple of (train_loader, val_loader, test_loader, extra_eval_loaders, saved_transform_config)
    """
    # Check if metadata has a 'split' column
    train_df = metadata[metadata['split'] == 'train'].copy()
    val_df = metadata[metadata['split'] == 'val'].copy()
    test_df = metadata[metadata['split'] == 'test'].copy()
        
    logger.info("Data splits: Train=%d, Val=%d, Test=%d", len(train_df), len(val_df), len(test_df))
    
    # Define transforms using centralized transform module
    # This ensures consistency between training and inference
    # Use transform_config if provided, otherwise use defaults
    if transform_config:
        train_transform, val_test_transform = create_transforms_from_config(transform_config, log_dir)
        # Use the provided transform_config
        saved_transform_config = transform_config
    else:
        # Fall back to default transforms
        train_transform, val_test_transform = create_transforms_from_config(DEFAULT_TRANSFORM_CONFIG, log_dir)
        saved_transform_config = DEFAULT_TRANSFORM_CONFIG
    
    # Create data loaders
    train_loader = create_data_loader(
        train_df,
        batch_size=train_config.batch_size,
        shuffle=True,
        transform=train_transform
    )
    
    val_loader = create_data_loader(
        val_df,
        batch_size=train_config.batch_size,
        shuffle=False,
        transform=val_test_transform
    )

    extra_eval_loaders = None
    if no_aug_train_loader:
        no_aug_train_df = train_df.sample(frac=0.1, random_state=random_seed)
        extra_eval_loaders = {
            "no_aug_train": create_data_loader(
                no_aug_train_df,
                batch_size=train_config.batch_size,
                shuffle=False,
                transform=val_test_transform,
            )
        }
    
    test_loader = create_data_loader(
        test_df,
        batch_size=train_config.batch_size,
        shuffle=False,
        transform=val_test_transform
    )
    
    return train_loader, val_loader, test_loader, extra_eval_loaders, saved_transform_config


def save_experiment_info(exp_dir: Path, config: TrainingConfig, metadata_path: str):
    """Save experiment information to JSON file."""
    info = {
        'experiment_dir': str(exp_dir),
        'created_at': datetime.now().isoformat(),
        'config': {
            'architecture': config.architecture,
            'training_mode': config.training_mode,
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'optimizer': config.optimizer,
            'loss_function': config.loss_function,
            'random_seed': config.random_seed,
        },
        'data': {
            'metadata_path': metadata_path,
        },
        'system': {
            'device': config.device
        }
    }
    
    with open(exp_dir / "experiment_info.json", 'w') as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train WSI ROI Detection model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config YAML file'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to metadata CSV file'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        required=True,
        help='Custom name for experiment directory'
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='experiments',
        help='Base directory for experiments'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.1,
        help='Test split ratio (default: 0.1)'
    )
    
    args = parser.parse_args()

    validate_training_paths(args.config, args.metadata)

    # Create experiment directory and setup the logger
    exp_dir = create_experiment_dir(args.experiment_dir, args.experiment_name)
    setup_logging(exp_dir)
    logger.info("Created experiment directory: %s", exp_dir)

    # Load config
    logger.info("Loading config from %s", args.config)
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    train_config = TrainingConfig(**config_dict)
    
    # Copy config to experiment directory
    config_copy_path = exp_dir / "configs" / "training_config.yaml"
    shutil.copy(args.config, config_copy_path)
    logger.info("Saved config to %s", config_copy_path)
    
    # Load metadata
    logger.info("Loading metadata from %s", args.metadata)
    metadata = load_metadata(args.metadata)
    logger.info("Loaded %d samples", len(metadata))
    
    # Prepare data loaders
    logger.info("Preparing data loaders...")
    # Get transform config from YAML; if missing, use DEFAULT_TRANSFORM_CONFIG (ImageNet norm, etc.)
    transform_config = config_dict.get('transforms', DEFAULT_TRANSFORM_CONFIG)
    train_loader, val_loader, test_loader, extra_eval_loaders, saved_transform_config = prepare_data_loaders(
        metadata,
        train_config,
        transform_config=transform_config,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=train_config.random_seed,
        log_dir=exp_dir / "logs",
        no_aug_train_loader=True
    )

    # Persist the resolved transform config as the experiment artifact.
    # Inference loads only this file—not the full training YAML—to build the inference transform.
    save_resolved_transform_config(saved_transform_config, exp_dir / "configs" / "transform_config.json")
    logger.info("Saved transform config to %s", exp_dir / "configs" / "transform_config.json")

    if extra_eval_loaders:
        logger.info("Extra evaluation loaders: %s", list(extra_eval_loaders.keys()))

    # Set save path for model
    model_save_path = exp_dir / "models" / "best_model.pth"
    train_config.save_dir = str(exp_dir / "models")
    train_config.checkpoint_dir = str(exp_dir / "checkpoints")
    
    # Save experiment info
    save_experiment_info(exp_dir, train_config, args.metadata)
    
    # Run training
    logger.info("")
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    save_paths = {
        'model': model_save_path,
        'history': exp_dir / "metrics" / "training_history.json",
        'artifacts': exp_dir / "artifacts"
    }
    
    results = train(
        config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        extra_eval_loaders=extra_eval_loaders,
        save_paths=save_paths,
        save_history_every_n_epochs=5,
    )

    # Save final model state
    final_model_path = exp_dir / "models" / "final_model.pth"
    torch.save(results['model'].state_dict(), final_model_path)
    logger.info("Saved final model to %s", final_model_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Training complete! Results saved to: %s", exp_dir)
    logger.info("=" * 60)
    logger.info("Best validation accuracy: %.4f", results['best_val_acc'])


if __name__ == "__main__":
    main()
