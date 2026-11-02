"""A starting point for a compact EEG recurrent classifier.

Keep this simple while learning the math. The recurrent forward pass is
intentionally left as an exercise for the project.
"""

import torch
from torch import nn


class AdaptiveRecurrentClassifier(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        samples: int = 300,
        hidden_size: int = 64,
        num_classes: int = 2,
        max_steps: int = 6,
    ):
        super().__init__()
        self.max_steps = max_steps

        # Input shape: [batch, channels, samples]
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * samples, hidden_size),
            nn.Tanh(),
        )

        # Input: [encoded features, previous hidden state]
        self.recurrent = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for every pass: [batch, max_steps, num_classes].

        TODO:
        1. Encode the input.
        2. Initialize a hidden state of zeros.
        3. Repeat the shared recurrent layer max_steps times.
        4. Predict with the classifier after each pass.
        5. Stack and return the predictions.

        Do not add early stopping yet. First verify that fixed-depth
        recurrence works and that predictions can improve across passes.
        """
        raise NotImplementedError("Implement the recurrent forward pass.")
