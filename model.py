"""Multi-exit Fashion-MNIST classifiers: the CNN used for the experiments and the earlier MLP."""

import math
from collections.abc import Callable

import torch
from torch import nn


# Multi-exit MLP (the first design; failed the exit-gap check).
DEFAULT_WIDTH = 32
DEFAULT_BLOCK_HIDDEN = 128
DEFAULT_NUM_EXITS = 10

# Multi-exit 2D CNN (the step-8 fallback adopted on November 22, 2026).
DEFAULT_CNN_CHANNELS = (16, 32)
DEFAULT_CNN_DOWNSAMPLE_AT = 4


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


class MultiExitNetwork(nn.Module):
    """A stem, then distinct blocks, each followed by its own classifier head.

    Exit k is the head after block k (exits are numbered from 1). Nothing is
    shared between blocks or between heads. Subclasses build self.blocks and
    self.heads and implement _stem.
    """

    input_shape: tuple[int, ...]
    num_exits: int
    num_classes: int
    blocks: nn.ModuleList
    heads: nn.ModuleList

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _check_input(self, x: torch.Tensor) -> None:
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected input shape [batch, {self.input_shape}], got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits from every exit: [batch, num_exits, num_classes]."""
        self._check_input(x)
        h = self._stem(x)
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
        self._check_input(x)
        h = self._stem(x)
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


class MultiExitClassifier(MultiExitNetwork):
    """Multi-exit MLP: 784->width projection, residual MLP blocks and linear heads.

    Failed the step-8 gate (flat accuracy across exits); kept for comparison.
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

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x.flatten(1))


class ConvBlock(nn.Module):
    """Residual 3x3 convolution with its own weights: shortcut(h) + Conv(ReLU(Norm(h))).

    With stride 2 or a change in channels, the shortcut is a 1x1 convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.shortcut = (
            nn.Identity() if stride == 1 and in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.shortcut(h) + self.conv(torch.relu(self.norm(h)))


def conv_head(channels: int, num_classes: int) -> nn.Sequential:
    """Normalize, average over the image, then a linear layer to class logits."""
    return nn.Sequential(
        nn.GroupNorm(1, channels), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(channels, num_classes),
    )


class MultiExitCNN(MultiExitNetwork):
    """Multi-exit 2D CNN.

    A 3x3 stride-2 stem (28x28 -> 14x14), residual 3x3 blocks with channels[0]
    channels, then from block downsample_at onward channels[1] channels at half
    the resolution (14x14 -> 7x7). Each block has its own pooled linear head.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...] = (1, 28, 28),
        channels: tuple[int, int] = DEFAULT_CNN_CHANNELS,
        downsample_at: int = DEFAULT_CNN_DOWNSAMPLE_AT,
        num_exits: int = DEFAULT_NUM_EXITS,
        num_classes: int = 10,
    ):
        super().__init__()
        if len(input_shape) != 3 or len(channels) != 2:
            raise ValueError("input_shape must be (channels, height, width) and channels must have two values")
        if min(*input_shape, *channels, num_exits, num_classes) < 1:
            raise ValueError("Input, channel, exit and class sizes must be positive")
        if not 2 <= downsample_at <= num_exits:
            raise ValueError("downsample_at must be between 2 and num_exits")
        self.input_shape = tuple(input_shape)
        self.num_exits = num_exits
        self.num_classes = num_classes
        narrow, wide = channels
        self.stem = nn.Conv2d(input_shape[0], narrow, 3, stride=2, padding=1)
        blocks, heads = [], []
        for number in range(1, num_exits + 1):
            if number < downsample_at:
                blocks.append(ConvBlock(narrow, narrow))
            elif number == downsample_at:
                blocks.append(ConvBlock(narrow, wide, stride=2))
            else:
                blocks.append(ConvBlock(wide, wide))
            heads.append(conv_head(narrow if number < downsample_at else wide, num_classes))
        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList(heads)

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


ARCHITECTURES = {"cnn": MultiExitCNN, "mlp": MultiExitClassifier}


def build_model(config: dict) -> MultiExitNetwork:
    """Rebuild a multi-exit network from a saved model_config.

    Configs saved before the CNN existed have no "architecture" key; they are MLPs.
    """
    settings = dict(config)
    architecture = settings.pop("architecture", "mlp")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}")
    if "channels" in settings:
        settings["channels"] = tuple(settings["channels"])
    return ARCHITECTURES[architecture](**settings)


def count_multiply_adds(model: MultiExitNetwork) -> dict:
    """Multiply-adds per image for the stem, each block and each head.

    Counts every use of a convolution or linear-layer weight while one image runs
    through the whole network (all blocks and all heads). Normalization,
    activations, pooling and bias additions are not counted. The result is
    {"stem": int, "blocks": [int per block], "heads": [int per head]}; everything
    not inside a block or head counts as the stem.
    """
    counts: dict[nn.Module, int] = {}

    def record(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            weights_per_output = module.in_channels // module.groups * math.prod(module.kernel_size)
        else:
            weights_per_output = module.in_features
        counts[module] = counts.get(module, 0) + output[0].numel() * weights_per_output

    layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    handles = [layer.register_forward_hook(record) for layer in layers]
    reference = next(model.parameters())
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros((1, *model.input_shape), dtype=reference.dtype, device=reference.device))
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()
    if set(counts) != set(layers):
        raise RuntimeError("Some convolution or linear layers did not run")

    def total(part: nn.Module) -> int:
        return int(sum(counts.get(m, 0) for m in part.modules()))

    blocks = [total(block) for block in model.blocks]
    heads = [total(head) for head in model.heads]
    return {"stem": total(model) - sum(blocks) - sum(heads), "blocks": blocks, "heads": heads}
