"""Small recurrent classifier for EEG motor imagery.

Each pass refines the representation of the same EEG window. Recurrence is
over computational depth, rather than consecutive EEG samples.
"""

import torch
from torch import nn


class AdaptiveRecurrentClassifier(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        samples: int = 301,
        hidden_size: int = 64,
        num_classes: int = 2,
        max_steps: int = 6,
    ):
        super().__init__()
        if min(channels, samples, hidden_size, num_classes, max_steps) < 1:
            raise ValueError("All model dimensions must be positive")
        self.channels = channels
        self.samples = samples
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_steps = max_steps

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * samples, hidden_size),
            nn.Tanh(),
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
