"""Small recurrent classifier for EEG motor imagery.

Each pass refines the representation of the same EEG window. Recurrence is
over computational depth, rather than consecutive EEG samples.
"""

import torch
from torch import nn


DEFAULT_HIDDEN_SIZE = 64
DEFAULT_MAX_STEPS = 12
DEFAULT_ENCODER = "eeg"
DEFAULT_DROPOUT = 0.25


class EEGEncoder(nn.Module):
    """EEGNet-inspired temporal, spatial, and separable filters for [B, C, T]."""

    def __init__(self, channels: int, hidden_size: int, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4), ceil_mode=True),
            nn.Dropout(dropout),
            nn.Conv2d(16, 16, (1, 15), padding=(0, 7), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8), ceil_mode=True),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4, hidden_size),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x.unsqueeze(1))


class AdaptiveRecurrentClassifier(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        samples: int = 301,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_classes: int = 2,
        max_steps: int = DEFAULT_MAX_STEPS,
        encoder_type: str = DEFAULT_ENCODER,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        if min(channels, samples, hidden_size, num_classes, max_steps) < 1:
            raise ValueError("All model dimensions must be positive")
        if encoder_type not in ("eeg", "dense"):
            raise ValueError("encoder_type must be 'eeg' or 'dense'")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.channels = channels
        self.samples = samples
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_steps = max_steps
        self.encoder_type = encoder_type
        self.dropout = dropout

        if encoder_type == "eeg":
            self.encoder = EEGEncoder(channels, hidden_size, dropout)
        else:
            # Preserve the parameter names of the original dense checkpoints.
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels * samples, hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
        self.recurrent = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def refine(self, encoded: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """One pass of the same recurrent computation block."""
        update = torch.tanh(self.recurrent(torch.cat([encoded, state], dim=-1)))
        return self.norm(state + update)

    def forward_fixed(self, x: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        """Logits at a chosen depth: [batch, num_classes]."""
        depth = self.max_steps if steps is None else steps
        if not 1 <= depth <= self.max_steps:
            raise ValueError("steps must be between 1 and max_steps")
        encoded = self.encoder(x)
        state = torch.zeros_like(encoded)
        for _ in range(depth):
            state = self.refine(encoded, state)
        return self.classifier(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits after every pass: [batch, max_steps, num_classes]."""
        encoded = self.encoder(x)
        state = torch.zeros_like(encoded)
        predictions = []
        for _ in range(self.max_steps):
            state = self.refine(encoded, state)
            predictions.append(self.classifier(state))
        return torch.stack(predictions, dim=1)

    @torch.no_grad()
    def predict_adaptive(
        self,
        x: torch.Tensor,
        confidence: float = 0.85,
        stability: float = 0.05,
        min_steps: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and steps used; remove halted examples from the batch."""
        if not 0 <= confidence <= 1 or not 0 <= stability <= 1:
            raise ValueError("confidence and stability must be between 0 and 1")
        if not 2 <= min_steps <= self.max_steps:
            raise ValueError("min_steps must be between 2 and max_steps")

        encoded = self.encoder(x)
        state = torch.zeros_like(encoded)
        active = torch.arange(x.shape[0], device=x.device)
        output = torch.empty((x.shape[0], self.num_classes), device=x.device, dtype=encoded.dtype)
        steps_taken = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        previous_probs = None
        previous_classes = None

        for step in range(1, self.max_steps + 1):
            state = self.refine(encoded, state)
            logits = self.classifier(state)
            probs = logits.softmax(dim=-1)
            predicted = probs.argmax(dim=-1)

            if previous_probs is None:
                finished = torch.zeros(len(active), device=x.device, dtype=torch.bool)
            else:
                difference = (probs - previous_probs).abs().amax(dim=-1)
                finished = (
                    (probs.amax(dim=-1) >= confidence)
                    & (difference <= stability)
                    & (predicted == previous_classes)
                    & (step >= min_steps)
                )
            if step == self.max_steps:
                finished = torch.ones_like(finished)

            output[active[finished]] = logits[finished]
            steps_taken[active[finished]] = step
            if step == self.max_steps:
                break

            remaining = ~finished
            active = active[remaining]
            if active.numel() == 0:
                break
            encoded = encoded[remaining]
            state = state[remaining]
            previous_probs = probs[remaining]
            previous_classes = predicted[remaining]

        return output, steps_taken


def model_from_checkpoint(checkpoint: dict) -> AdaptiveRecurrentClassifier:
    """Load current models and original dense-encoder checkpoints."""
    config = dict(checkpoint["model_config"])
    if "encoder_type" not in config:
        config.update(encoder_type="dense", dropout=0.0)
    model = AdaptiveRecurrentClassifier(**config)
    model.load_state_dict(checkpoint["model_state"])
    return model
