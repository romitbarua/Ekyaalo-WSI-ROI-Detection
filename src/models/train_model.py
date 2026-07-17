import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from pathlib import Path
from typing import Dict, Optional, Any, Union, Tuple
import yaml, json
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
    architecture: str = 'resnet18'
    num_classes: int = 2
    pretrained: bool = True
    model_params: Optional[Dict[str, Any]] = None
    training_mode: str = 'fine_tune'
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    optimizer_params: Optional[Dict[str, Any]] = None
    loss_function: str = 'cross_entropy'
    label_smoothing: float = 0.0
    max_grad_norm: float = 0
    early_stop_patience: int = 0
    transforms: Dict[str, Any] = None
    k_folds: int = 5
    random_seed: int = 42
    data_path: Optional[str] = None
    save_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    device: Optional[str] = None

    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {}
        if self.optimizer_params is None:
            self.optimizer_params = {}
        if self.device is None:
            self.device = get_device_preference()


def setup_model(config: TrainingConfig) -> nn.Module:
    return load_model_from_config(
        architecture=config.architecture, num_classes=config.num_classes,
        pretrained=config.pretrained, model_params=config.model_params,
        device=config.device,
    )


def _unfreeze_head(model: nn.Module):
    """Find and unfreeze the classifier head (fc / classifier / head / last child)."""
    for attr in ('fc', 'classifier', 'head'):
        if hasattr(model, attr):
            for p in getattr(model, attr).parameters():
                p.requires_grad = True
            return
    last = list(model.children())[-1]
    if isinstance(last, (nn.Linear, nn.Sequential)):
        for p in last.parameters():
            p.requires_grad = True
    else:
        logger.warning("Couldn't find head, training everything")
        for p in model.parameters():
            p.requires_grad = True


def configure_training_mode(model: nn.Module, mode: str) -> None:
    if mode == 'freeze':
        for p in model.parameters():
            p.requires_grad = False
    elif mode == 'head_only':
        for p in model.parameters():
            p.requires_grad = False
        _unfreeze_head(model)
    elif mode == 'fine_tune':
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown training mode: {mode}")
    logger.info("Training mode: %s", mode)


def get_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    cls = {'adam': optim.Adam, 'adamw': optim.AdamW, 'sgd': optim.SGD}.get(config.optimizer.lower())
    if cls is None:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    return cls(params, lr=config.learning_rate, **config.optimizer_params)


def compute_class_weights(loader: DataLoader, n_classes: int, device: str) -> torch.Tensor:
    counts = torch.zeros(n_classes)
    for _, labels in loader:
        for c in labels:
            counts[c.item()] += 1
    w = counts.sum() / (n_classes * counts.clamp(min=1))
    logger.info("Class weights: %s (counts %s)", w.tolist(), counts.int().tolist())
    return w.to(device)


def get_loss_function(config: TrainingConfig, class_weights: torch.Tensor = None) -> nn.Module:
    if config.loss_function.lower() in ('cross_entropy', 'focal'):
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
    raise ValueError(f"Unknown loss: {config.loss_function}")


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=0):
    model.train()
    total = correct = 0
    running_loss = 0.0
    trainable = [p for p in model.parameters() if p.requires_grad]
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        total += labels.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        running_loss += loss.item() * labels.size(0)
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total = correct = 0
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            running_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (running_loss / total, correct / total,
            np.array(all_labels), np.array(all_preds), np.array(all_probs))


def compute_metrics(labels, preds, probs):
    m = dict(
        accuracy=accuracy_score(labels, preds),
        precision=precision_score(labels, preds, average='weighted', zero_division=0),
        recall=recall_score(labels, preds, average='weighted', zero_division=0),
        f1=f1_score(labels, preds, average='weighted', zero_division=0),
        per_class_acc=recall_score(labels, preds, average=None, zero_division=0).tolist(),
    )
    if probs.shape[1] == 2:
        m['auc'] = roc_auc_score(labels, probs[:, 1])
    return m


def train(config, train_loader, val_loader, save_paths,
          extra_eval_loaders=None, save_history_every_n_epochs=5):

    if isinstance(config, (str, Path)):
        p = Path(config)
        with open(p) as f:
            raw = yaml.safe_load(f) if p.suffix in ('.yaml', '.yml') else json.load(f)
        config = TrainingConfig(**raw)
    elif isinstance(config, dict):
        config = TrainingConfig(**config)

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    device = torch.device(config.device)
    model_save_path = save_paths['model']
    history_save_path = save_paths['history']
    artifact_save_path = save_paths['artifacts']

    model = setup_model(config)

    # flops
    model.eval()
    inp_shape = train_loader.dataset[0][0].shape
    fc = FlopCounterMode(mods=model, display=False, depth=None)
    with fc:
        model(torch.randn((1,) + inp_shape).to(device))
    flops = fc.get_total_flops()
    logger.info("FLOPS: %s", flops)
    model.train()

    configure_training_mode(model, config.training_mode)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Params: %s / %s trainable", f"{n_train:,}", f"{n_total:,}")

    class_weights = compute_class_weights(train_loader, config.num_classes, device)
    criterion = get_loss_function(config, class_weights=class_weights)
    optimizer = get_optimizer(model, config)

    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)
    warmup_epochs = min(5, config.num_epochs // 5)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_per_class_acc = []
    stale = 0
    history = dict(flops=flops, train_loss=[], train_acc=[], val_loss=[], val_acc=[], val_per_class_acc=[], metrics=[])
    if extra_eval_loaders:
        for name in extra_eval_loaders:
            history.update({f'{name}_loss': [], f'{name}_acc': [], f'{name}_metrics': []})

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=config.max_grad_norm,
        )
        val_loss, val_acc, val_labels, val_preds, val_probs = validate(model, val_loader, criterion, device)

        extra_results = {}
        if extra_eval_loaders:
            for name, loader in extra_eval_loaders.items():
                l, a, lb, pr, pb = validate(model, loader, criterion, device)
                extra_results[name] = (l, a, compute_metrics(lb, pr, pb))

        metrics = compute_metrics(val_labels, val_preds, val_probs)
        val_auc = metrics.get('auc', 0)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_per_class_acc'].append(metrics.get('per_class_acc', []))
        history['metrics'].append(metrics)
        for name, (l, a, m) in extra_results.items():
            history[f'{name}_loss'].append(l)
            history[f'{name}_acc'].append(a)
            history[f'{name}_metrics'].append(m)

        scheduler.step()

        logger.info(
            "Epoch %02d/%d | Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f | AUC: %.4f | LR: %.6f",
            epoch, config.num_epochs, train_loss, train_acc, val_loss, val_acc,
            val_auc, scheduler.get_last_lr()[0],
        )

        # track best by AUC (acc is misleading with class imbalance)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_val_per_class_acc = metrics.get('per_class_acc', [])
            stale = 0
            if model_save_path:
                mp = Path(model_save_path)
                mp.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), mp)
                logger.info("Saved best model (auc=%.4f)", val_auc)
        else:
            stale += 1

        if history_save_path and (epoch % save_history_every_n_epochs == 0 or epoch == config.num_epochs):
            hp = Path(history_save_path)
            hp.parent.mkdir(parents=True, exist_ok=True)
            hp.write_text(json.dumps(dict(
                history=history, best_val_auc=best_val_auc,
                best_val_acc=best_val_acc,
                best_val_per_class_acc=best_val_per_class_acc,
                config=asdict(config),
            ), indent=2))
            if artifact_save_path:
                plot_loss_curve(history, artifact_save_path)
                plot_accuracy_curve(history=history, save_dir=artifact_save_path, extra_eval_loaders=extra_eval_loaders)

        if config.early_stop_patience > 0 and stale >= config.early_stop_patience:
            logger.info("Early stop at epoch %d", epoch)
            break

    logger.info("Done. best auc=%.4f acc=%.4f per_class=%s", best_val_auc, best_val_acc, best_val_per_class_acc)
    return dict(model=model, history=history, best_val_acc=best_val_acc, best_val_auc=best_val_auc,
                best_val_per_class_acc=best_val_per_class_acc, config=asdict(config))
