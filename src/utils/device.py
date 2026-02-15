"""
Device resolution for training and inference.

Uses a single preference order: CUDA → MPS (Apple Silicon) → CPU.
Allows config override (e.g. device: "cpu") for reproducibility or debugging.
"""

import torch
from typing import Optional, Union


def get_device_preference() -> str:
    """
    Return the preferred device string when no override is set.
    Order: CUDA → MPS (if available) → CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device(prefer: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolve the device to use for model and tensors.

    Args:
        prefer: Optional override. If set (e.g. from config), that device is used.
                Otherwise auto-detects: CUDA → MPS → CPU.

    Returns:
        torch.device to use.
    """
    if prefer is not None:
        if isinstance(prefer, torch.device):
            return prefer
        return torch.device(prefer)
    return torch.device(get_device_preference())
