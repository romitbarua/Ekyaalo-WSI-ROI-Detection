"""
Baseline CNN architecture for binary classification.

A lightweight, simple CNN architecture that can be used as an alternative
to pretrained models. Useful for baseline comparisons or when computational
resources are limited.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    A simple baseline CNN architecture for binary classification.
    
    Architecture:
    - 3-4 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
    - Global Average Pooling
    - Fully connected layers for classification
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3,
        base_channels: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.5
    ):
        """
        Initialize the Baseline CNN.
        
        Args:
            num_classes: Number of output classes (default: 2)
            input_channels: Number of input channels (default: 3 for RGB)
            base_channels: Number of channels in the first conv layer (default: 32)
            num_blocks: Number of convolutional blocks (default: 4)
            dropout: Dropout probability for fully connected layers (default: 0.5)
        """
        super(BaselineCNN, self).__init__()
        
        self.num_blocks = num_blocks
        self.base_channels = base_channels
        
        # Build convolutional blocks
        conv_blocks = []
        in_channels = input_channels
        
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            conv_blocks.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*conv_blocks)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_baseline_cnn(
    device: torch.device,
    num_classes: int = 2,
    input_channels: int = 3,
    base_channels: int = 32,
    num_blocks: int = 4,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Create a baseline CNN model for binary classification.

    Caller must resolve and pass device (e.g. via get_device() or config).
    
    Args:
        device: Device to move the model to (required; caller owns device resolution)
        num_classes: Number of output classes (default: 2)
        input_channels: Number of input channels (default: 3 for RGB)
        base_channels: Number of channels in the first conv layer (default: 32)
                      This will double with each block (32, 64, 128, 256, ...)
        num_blocks: Number of convolutional blocks (default: 4)
                   More blocks = deeper network = more parameters
        dropout: Dropout probability for fully connected layers (default: 0.5)
    
    Returns:
        Baseline CNN model ready for training/inference
    
    Examples:
        >>> device = get_device()
        >>> model = create_baseline_cnn(num_classes=2, device=device)
    """
    model = BaselineCNN(
        num_classes=num_classes,
        input_channels=input_channels,
        base_channels=base_channels,
        num_blocks=num_blocks,
        dropout=dropout
    )
    
    # Move model to device
    model = model.to(device)
    
    return model
