"""Fashion-MNIST classifiers: the multi-exit network and the single-exit DNN baseline."""

import math
from collections.abc import Callable

import torch
from torch import nn


DEFAULT_HIDDEN_SIZES = (256, 128)
DEFAULT_DROPOUT = 0.2

# Adopted multi-exit design (IMPLEMENTATION_PLAN.md, decision 1.1).
DEFAULT_WIDTH = 32
DEFAULT_BLOCK_HIDDEN = 128
DEFAULT_NUM_EXITS = 10


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


class ResidualBlock(nn.Module):
    """One block with its own weights: h + W2 * ReLU(W1 * LayerNorm(h))."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.expand = nn.Linear(width, hidden)
        self.contract = nn.Linear(hidden, width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.contract(torch.relu(self.expand(self.norm(h))))


# should_stop(exit_number, logits_so_far) -> bool tensor, one entry per running image.
StopRule = Callable[[int, torch.Tensor], torch.Tensor]


class MultiExitClassifier(nn.Module):
    """Projection, then distinct residual blocks, each followed by its own classifier head.

    Exit k is the head after block k (exits are numbered from 1). Nothing is
    shared between blocks or between heads.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...] = (1, 28, 28),
        width: int = DEFAULT_WIDTH,
        block_hidden: int = DEFAULT_BLOCK_HIDDEN,
        num_exits: int = DEFAULT_NUM_EXITS,
        num_classes: int = 10,
    ):
        super().__init__()
        if not input_shape or min(*input_shape, width, block_hidden, num_exits, num_classes) < 1:
            raise ValueError("Input, width, block, exit and class sizes must be positive")
        self.input_shape = tuple(input_shape)
        self.num_exits = num_exits
        self.num_classes = num_classes
        self.projection = nn.Linear(math.prod(input_shape), width)
        self.blocks = nn.ModuleList(ResidualBlock(width, block_hidden) for _ in range(num_exits))
        self.heads = nn.ModuleList(nn.Linear(width, num_classes) for _ in range(num_exits))

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected input shape [batch, {self.input_shape}], got {tuple(x.shape)}")
        return self.projection(x.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits from every exit: [batch, num_exits, num_classes]."""
        h = self._project(x)
        logits = []
        for block, head in zip(self.blocks, self.heads):
            h = block(h)
            logits.append(head(h))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def predict_sequential(
        self, x: torch.Tensor, should_stop: StopRule, min_exit: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run exits one at a time and stop each image as soon as should_stop says so.

        should_stop(exit_number, logits_so_far) receives the 1-based exit number and
        the logits of every exit computed so far for the images still running,
        shaped [running, exit_number, num_classes]. It returns a boolean tensor
        with one entry per running image. It is consulted only from min_exit
        onward, and every image still running stops at the last exit. Stopped
        images are removed from the batch, so later blocks run only for the rest.

        Returns the logits of the exit each image stopped at, [batch, num_classes],
        and that exit's number, [batch].
        """
        if not 1 <= min_exit <= self.num_exits:
            raise ValueError("min_exit must be between 1 and num_exits")
        h = self._project(x)
        running = torch.arange(len(x), device=x.device)
        history = h.new_empty((len(x), 0, self.num_classes))
        chosen = h.new_empty((len(x), self.num_classes))
        exit_numbers = torch.zeros(len(x), dtype=torch.long, device=x.device)
        for number, (block, head) in enumerate(zip(self.blocks, self.heads), start=1):
            h = block(h)
            history = torch.cat([history, head(h)[:, None]], dim=1)
            if number == self.num_exits:
                stop = torch.ones(len(running), dtype=torch.bool, device=x.device)
            elif number < min_exit:
                stop = torch.zeros(len(running), dtype=torch.bool, device=x.device)
            else:
                stop = torch.as_tensor(should_stop(number, history), dtype=torch.bool, device=x.device)
                if stop.shape != (len(running),):
                    raise ValueError(f"should_stop must return {len(running)} values, got {tuple(stop.shape)}")
            chosen[running[stop]] = history[stop, -1]
            exit_numbers[running[stop]] = number
            running, h, history = running[~stop], h[~stop], history[~stop]
            if len(running) == 0:
                break
        return chosen, exit_numbers
