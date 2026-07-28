"""
Scripts to create exploratory and results-oriented visualizations.

This module contains functions for:
- ROC curves and confusion matrices
- Model performance plots
- Interactive heatmaps
- Comparative visualizations
"""

import os
from typing import Optional
import matplotlib.pyplot as plt

def plot_loss_curve(history: dict, save_dir: Optional[str] = None) -> None:
    """
    Plot and save loss curve from training history.

    Args:
        history (dict): Training history.
        save_dir (str, optional): Directory to save the plot.
    """
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_accuracy_curve(history: dict, save_dir: Optional[str] = None, extra_eval_loaders: Optional[dict] = None) -> None:
    """
    Plot and save accuracy curve from training history.

    Args:
        history (dict): Training history.
        save_dir (str, optional): Directory to save the plot.
    """
    plt.figure()
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    if extra_eval_loaders:
        for name, _ in extra_eval_loaders.items():
            plt.plot(history[f'{name}_acc'], label=f'{name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "accuracy_curve.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()