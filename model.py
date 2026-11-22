"""Feedforward DNN for Fashion-MNIST image classification."""

import math

import torch
from torch import nn


DEFAULT_HIDDEN_SIZES = (256, 128)
DEFAULT_DROPOUT = 0.2


class DNNClassifier(nn.Module):
    """Flatten an image and classify it with independently learned dense layers."""

    def __init__(
        self,
        input_shape: tuple[int, ...] = (1, 28, 28),
        hidden_sizes: tuple[int, ...] = DEFAULT_HIDDEN_SIZES,
        num_classes: int = 10,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        if not input_shape or not hidden_sizes or min(*input_shape, *hidden_sizes, num_classes) < 1:
            raise ValueError("Input, hidden and output dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive)")
        self.input_shape = tuple(input_shape)
        self.num_classes = num_classes
        layers = [nn.Flatten()]
        width = math.prod(input_shape)
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(width, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            width = hidden_size
        layers.append(nn.Linear(width, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one set of class logits per image: [batch, num_classes]."""
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected input shape [batch, {self.input_shape}], got {tuple(x.shape)}")
        return self.network(x)
