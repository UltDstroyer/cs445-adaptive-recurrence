"""Load and split BCI Competition IV Dataset 2b motor-imagery EEG trials.

MOABB filters and epochs public EEG recordings. Only training data is used to
calculate the channel means and standard deviations for normalization.
"""

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class DataSplits:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    session_names: list[str]


def load_trials(subject: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trials [N, C, T], class labels 0/1, and one session ID per trial."""
    from moabb.datasets import BNCI2014_004
    from moabb.paradigms import LeftRightImagery

    dataset = BNCI2014_004()
    paradigm = LeftRightImagery(fmin=8, fmax=32, resample=100, tmin=0.5, tmax=3.5)
    x, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    mapping = {"left_hand": 0, "right_hand": 1}
    unknown = set(labels) - set(mapping)
    if unknown:
        raise ValueError(f"Unexpected EEG labels: {sorted(unknown)}")
    y = np.array([mapping[label] for label in labels], dtype=np.int64)
    sessions = metadata["session"].astype(str).to_numpy()
    if len(x) != len(y) or len(y) != len(sessions) or x.shape[1] != 3:
        raise ValueError(f"Unexpected EEG shapes: x={x.shape}, labels={y.shape}")
    return x, y, sessions


def demo_trials(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic signals for an end-to-end code test; NOT real EEG."""
    rng = np.random.default_rng(seed)
    n_per_session = 48
    samples = 301
    t = np.arange(samples, dtype=np.float32) / 100.0
    x = rng.normal(0, 0.45, (5 * n_per_session, 3, samples)).astype(np.float32)
    y = np.tile(np.arange(n_per_session) % 2, 5).astype(np.int64)
    for i, label in enumerate(y):
        x[i, 0 if label == 0 else 2] += 1.5 * np.sin(
            2 * np.pi * (12 if label == 0 else 20) * t
        )
    sessions = np.repeat(["0train", "1train", "2train", "3test", "4test"], n_per_session)
    return x, y, sessions


def split_trials(
    x: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> DataSplits:
    """First 3 sessions=train; fourth=validation; fifth=held-out test."""
    names = sorted(set(map(str, sessions)))
    if len(names) != 5:
        raise ValueError(f"Expected five sessions, found {names}")
    train_mask = np.isin(sessions, names[:3])
    val_mask = sessions == names[3]
    test_mask = sessions == names[4]
    train_x, val_x, test_x = x[train_mask], x[val_mask], x[test_mask]
    train_y, val_y, test_y = y[train_mask], y[val_mask], y[test_mask]
    if min(len(train_x), len(val_x), len(test_x)) == 0:
        raise ValueError("At least one dataset split is empty")
    if stats is None:
        mean = train_x.mean(axis=(0, 2), keepdims=True)
        std = np.maximum(train_x.std(axis=(0, 2), keepdims=True), 1e-7)
    else:
        mean, std = stats
        mean = np.asarray(mean, dtype=np.float32).reshape(1, x.shape[1], 1)
        std = np.asarray(std, dtype=np.float32).reshape(1, x.shape[1], 1)
        if np.any(std <= 0):
            raise ValueError("Invalid channel standard deviations")

    def normalize(a: np.ndarray) -> np.ndarray:
        return ((a - mean) / std).astype(np.float32)

    return DataSplits(
        normalize(train_x), train_y, normalize(val_x), val_y,
        normalize(test_x), test_y, mean.astype(np.float32), std.astype(np.float32), names,
    )


def get_splits(
    subject: int = 1,
    demo: bool = False,
    stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> DataSplits:
    x, y, sessions = demo_trials() if demo else load_trials(subject)
    return split_trials(x, y, sessions, stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--demo", action="store_true", help="Synthetic data, not EEG")
    args = parser.parse_args()
    splits = get_splits(args.subject, args.demo)
    print("SYNTHETIC DEMO DATA" if args.demo else "BCI Competition IV Dataset 2b")
    print("Sessions:", splits.session_names)
    for name in ("train", "val", "test"):
        x, y = getattr(splits, f"{name}_x"), getattr(splits, f"{name}_y")
        print(f"{name:5s}: {x.shape}, class counts={np.bincount(y, minlength=2).tolist()}")


if __name__ == "__main__":
    main()
