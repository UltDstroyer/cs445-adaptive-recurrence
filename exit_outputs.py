"""Saved per-exit outputs and the metrics computed from them.

evaluate.py runs a trained multi-exit network once over a split and saves every
exit's logits with the labels. Everything after that (threshold sweeps, stopping
rules, calibration) works from these files with numpy alone: no PyTorch, no
model and no access to the images.

File format (.npz, loaded with allow_pickle=False):
    logits         float32 [images, exits, classes]
    labels         int64   [images]
    image_indices  int64   [images]  position in the official partition
                                     (training partition for val, test partition for test)
    metadata       JSON string: split, checkpoint, model and data settings, ...
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


@dataclass
class ExitOutputs:
    logits: np.ndarray
    labels: np.ndarray
    image_indices: np.ndarray
    metadata: dict

    @property
    def num_exits(self) -> int:
        return self.logits.shape[1]

    @property
    def num_classes(self) -> int:
        return self.logits.shape[2]

    def probabilities(self) -> np.ndarray:
        return softmax(self.logits)


def _check(logits: np.ndarray, labels: np.ndarray, image_indices: np.ndarray) -> None:
    if logits.ndim != 3 or len(logits) == 0:
        raise ValueError("logits must be shaped [images, exits, classes]")
    if labels.shape != (len(logits),) or image_indices.shape != (len(logits),):
        raise ValueError("labels and image_indices need one entry per image")
    if labels.min() < 0 or labels.max() >= logits.shape[2]:
        raise ValueError("labels must be class numbers below the number of classes")
    if len(np.unique(image_indices)) != len(image_indices):
        raise ValueError("image_indices must not repeat")
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")


def save_exit_outputs(path: Path, logits, labels, image_indices, metadata: dict) -> None:
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    image_indices = np.asarray(image_indices, dtype=np.int64)
    _check(logits, labels, image_indices)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle, logits=logits, labels=labels, image_indices=image_indices,
            metadata=np.array(json.dumps({"format_version": FORMAT_VERSION, **metadata})),
        )


def load_exit_outputs(path: Path) -> ExitOutputs:
    with np.load(path, allow_pickle=False) as saved:
        outputs = ExitOutputs(
            logits=saved["logits"], labels=saved["labels"],
            image_indices=saved["image_indices"], metadata=json.loads(str(saved["metadata"])),
        )
    if outputs.metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} has an unsupported format version")
    _check(outputs.logits, outputs.labels, outputs.image_indices)
    return outputs


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float64)
    return exp / exp.sum(axis=-1, keepdims=True)


def macro_f1(labels: np.ndarray, predicted: np.ndarray, num_classes: int) -> float:
    """Mean of per-class F1; a class with no true or predicted images counts as 0."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(
        2 * confusion.diagonal(), denominator,
        out=np.zeros(num_classes, dtype=np.float64), where=denominator > 0,
    )
    return float(f1.mean())


def exit_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Accuracy, macro-F1 and cross-entropy for every exit."""
    num_classes = logits.shape[2]
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    exit_loss = -log_probs[np.arange(len(labels)), :, labels].mean(axis=0)
    predicted = logits.argmax(axis=-1)
    return {
        "exit_accuracy": [float((predicted[:, k] == labels).mean()) for k in range(logits.shape[1])],
        "exit_macro_f1": [macro_f1(labels, predicted[:, k], num_classes) for k in range(logits.shape[1])],
        "exit_loss": [float(v) for v in exit_loss],
    }
