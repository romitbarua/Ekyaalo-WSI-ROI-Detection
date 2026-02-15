"""
Flexible training script for WSI ROI detection models.

This module provides config-based training with support for:
- Multiple architectures (ResNet-18, EfficientNet-B0, MobileNet-V3, Baseline-CNN)
- Different training modes (freeze, head-only, fine-tune)
- Configurable model parameters
- K-fold cross-validation
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Any, Union, Tuple
import yaml
import json
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc
)

from .architectures import get_available_architectures
from .load_model import load_model_from_config
from src.utils.device import get_device_preference
from src.visualization.visualize import plot_loss_curve, plot_accuracy_curve

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model configuration
    architecture: str = 'resnet18'
    num_classes: int = 2
    pretrained: bool = True
    model_params: Optional[Dict[str, Any]] = None
    
    # Training mode: 'freeze', 'head_only', 'fine_tune'
    training_mode: str = 'fine_tune'
    
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = 'adam'  # 'adam' or 'sgd'
    optimizer_params: Optional[Dict[str, Any]] = None
    loss_function: str = 'cross_entropy'  # 'cross_entropy' or 'focal'

    # Training Transforms
    transforms: Dict[str, Any] = None
    
    # K-fold cross-validation
    k_folds: int = 5
    random_seed: int = 42
    
    # Paths
    data_path: Optional[str] = None
    save_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    
    # Device
    device: Optional[str] = None  # 'cuda', 'mps', 'cpu', or None for auto-detect
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.model_params is None:
            self.model_params = {}
        if self.optimizer_params is None:
            self.optimizer_params = {}
        if self.device is None:
            self.device = get_device_preference()


def setup_model(config: TrainingConfig) -> nn.Module:
    """Create model from training config."""
    return load_model_from_config(
        architecture=config.architecture,
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        model_params=config.model_params,
        device=config.device
    )


def configure_training_mode(model: nn.Module, mode: str) -> None:
    """
    Configure which parts of the model to train.
    
    Args:
        model: The model to configure
        mode: Training mode - 'freeze', 'head_only', or 'fine_tune'
    """
    if mode == 'freeze':
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        logger.info("All model parameters frozen.")
        
    elif mode == 'head_only':
        # Freeze backbone, only train classifier head
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the classifier/head
        # This works for most architectures (ResNet, EfficientNet, MobileNet)
        if hasattr(model, 'fc'):  # ResNet
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):  # EfficientNet, MobileNet
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'):  # Some timm models
            for param in model.head.parameters():
                param.requires_grad = True
        else:
            # Try to find the last layer
            last_layer = list(model.children())[-1]
            if isinstance(last_layer, nn.Linear) or isinstance(last_layer, nn.Sequential):
                for param in last_layer.parameters():
                    param.requires_grad = True
            else:
                logger.warning("Could not find classifier head. Training all parameters.")
                for param in model.parameters():
                    param.requires_grad = True
        
        logger.info("Only classifier head will be trained.")
        
    elif mode == 'fine_tune':
        # Train all parameters
        for param in model.parameters():
            param.requires_grad = True
        logger.info("All model parameters will be fine-tuned.")
        
    else:
        raise ValueError(
            f"Unknown training mode: {mode}. "
            f"Must be one of: 'freeze', 'head_only', 'fine_tune'"
        )


def get_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """
    Create optimizer based on config.
    
    Args:
        model: The model to optimize
        config: Training configuration
        
    Returns:
        Optimizer instance
    """
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            trainable_params,
            lr=config.learning_rate,
            **config.optimizer_params
        )
    elif config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            trainable_params,
            lr=config.learning_rate,
            **config.optimizer_params
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}. Must be 'adam' or 'sgd'")
    
    return optimizer


def get_loss_function(config: TrainingConfig) -> nn.Module:
    """
    Get loss function based on config.
    
    Args:
        config: Training configuration
        
    Returns:
        Loss function
    """
    if config.loss_function.lower() == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif config.loss_function.lower() == 'focal':
        # Focal loss implementation (if needed, can be added)
        # For now, fall back to cross entropy
        logger.warning("Focal loss not implemented, using CrossEntropyLoss")
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(
            f"Unknown loss function: {config.loss_function}. "
            f"Must be 'cross_entropy' or 'focal'"
        )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total = 0
    correct = 0
    running_loss = 0.0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        running_loss += loss.item() * labels.size(0)
    
    acc = correct / total
    avg_loss = running_loss / total
    return avg_loss, acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (average_loss, accuracy, all_labels, all_preds, all_probs)
    """
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            running_loss += loss.item() * labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = correct / total
    avg_loss = running_loss / total
    
    return (
        avg_loss,
        acc,
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        labels: True labels
        preds: Predictions
        probs: Prediction probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds, average='weighted', zero_division=0),
        'f1': f1_score(labels, preds, average='weighted', zero_division=0),
    }
    
    # AUC for binary classification
    if probs.shape[1] == 2:
        metrics['auc'] = roc_auc_score(labels, probs[:, 1])
    
    return metrics


def train(
    config: Union[TrainingConfig, Dict, str, Path],
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_paths: Dict[str, Union[str, Path]],
    extra_eval_loaders: Optional[Dict[str, DataLoader]] = None,
    save_path: Optional[Union[str, Path]] = None,
    history_save_path: Optional[Union[str, Path]] = None,
    save_history_every_n_epochs: int = 5,
) -> Dict[str, Any]:
    """
    Train a model based on configuration.

    Args:
        config: Training configuration (TrainingConfig, dict, or path to YAML/JSON)
        train_loader: Training data loader
        val_loader: Validation data loader
        save_paths: Paths for model, history, artifacts
        extra_eval_loaders: Optional dict of name -> DataLoader for extra evaluation (e.g. train without aug). Logged as {name}_loss, {name}_acc, {name}_metrics.
        save_path: Path to save the trained model (optional)
        history_save_path: Path to save training_history.json (optional). If set, written every save_history_every_n_epochs and at end.
        save_history_every_n_epochs: Write history to disk every N epochs (default 5).

    Returns:
        Dictionary containing training history and metrics
    """
    # Load config if needed
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unknown config file format: {config_path.suffix}")
        config = TrainingConfig(**config_dict)
    elif isinstance(config, dict):
        config = TrainingConfig(**config)
    elif not isinstance(config, TrainingConfig):
        raise TypeError(f"config must be TrainingConfig, dict, or path, got {type(config)}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    
    device = torch.device(config.device)

    model_save_path = save_paths['model']
    history_save_path = save_paths['history']
    artifact_save_path = save_paths['artifacts']
    
    # Create model
    logger.info("Creating %s model...", config.architecture)
    model = setup_model(config)
    
    # Configure training mode
    logger.info("Training mode: %s", config.training_mode)
    configure_training_mode(model, config.training_mode)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Trainable parameters: %s / %s", f"{trainable_params:,}", f"{total_params:,}")
    
    # Setup loss and optimizer
    criterion = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    
    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'metrics': []
    }
    if extra_eval_loaders:
        for name in extra_eval_loaders:
            history[f'{name}_loss'] = []
            history[f'{name}_acc'] = []
            history[f'{name}_metrics'] = []
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_labels, val_preds, val_probs = validate(
            model, val_loader, criterion, device
        )
        # Given the augmentations, we want to know how the model performs on the training set w/o augmentation
        extra_results = {}
        if extra_eval_loaders:
            for name, loader in extra_eval_loaders.items():
                loss, acc, labels, preds, probs = validate(model, loader, criterion, device)
                extra_results[name] = (loss, acc, compute_metrics(labels, preds, probs))
        
        # Compute metrics
        metrics = compute_metrics(val_labels, val_preds, val_probs)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['metrics'].append(metrics)
        for name, (loss, acc, m) in extra_results.items():
            history[f'{name}_loss'].append(loss)
            history[f'{name}_acc'].append(acc)
            history[f'{name}_metrics'].append(m)
        
        # Log progress
        logger.info(
            "Epoch %02d/%d | Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f | AUC: %.4f",
            epoch, config.num_epochs, train_loss, train_acc, val_loss, val_acc, metrics.get('auc', 0)
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                logger.info("Saved best model (val_acc: %.4f) to %s", val_acc, save_path)

        # Save history every N epochs (and on last epoch) for recovery if job stops
        if history_save_path and (
            epoch % save_history_every_n_epochs == 0 or epoch == config.num_epochs
        ):
            path = Path(history_save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(
                    {
                        "history": history,
                        "best_val_acc": best_val_acc,
                        "config": asdict(config),
                    },
                    f,
                    indent=2,
                )
            logger.info("Saved history to %s (epoch %d)", path, epoch)

            # save the plots
            if artifact_save_path:
                plot_loss_curve(history, artifact_save_path)
                plot_accuracy_curve(history=history, save_dir=artifact_save_path, extra_eval_loaders=extra_eval_loaders)
                
                logger.info("Saved plots to %s", artifact_save_path)
    
    logger.info("Training complete! Best validation accuracy: %.4f", best_val_acc)
    
    return {
        'model': model,
        'history': history,
        'best_val_acc': best_val_acc,
        'config': asdict(config)
    }


def main():
    """Main function for model training."""
    # Example usage
    logger.info("Training script ready!")
    logger.info("Available architectures: %s", get_available_architectures())
    logger.info("Example config:")
    example_config = TrainingConfig(
        architecture='resnet18',
        training_mode='head_only',
        num_epochs=10,
        batch_size=32
    )
    logger.info("\n%s", yaml.dump(asdict(example_config), default_flow_style=False))


if __name__ == "__main__":
    main()
