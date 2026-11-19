"""Fashion-MNIST images for the DNN and legacy EEG loading helpers."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FASHION_CLASSES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
DEFAULT_DATA_DIR = Path("data/vision")


@dataclass
class ImageSplits:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    mean: float
    std: float
    class_names: tuple[str, ...] = FASHION_CLASSES


def load_fashion_mnist(data_dir: Path, download: bool = True):
    """Read the official train/test partitions, downloading into data_dir."""
    from torchvision.datasets import FashionMNIST

    # Use the publisher's HTTPS mirror rather than torchvision's older HTTP URL.
    class HTTPSFashionMNIST(FashionMNIST):
        mirrors = [
            "https://raw.githubusercontent.com/zalandoresearch/"
            "fashion-mnist/master/data/fashion/"
        ]

        @property
        def raw_folder(self):
            return str(Path(self.root) / "FashionMNIST" / "raw")

    train = HTTPSFashionMNIST(root=str(data_dir), train=True, download=download)
    test = HTTPSFashionMNIST(root=str(data_dir), train=False, download=download)
    return train.data.numpy(), train.targets.numpy(), test.data.numpy(), test.targets.numpy()


def demo_images(seed: int = 7):
    """Artificial ten-class images for offline checks, not Fashion-MNIST data."""
    rng = np.random.default_rng(seed)
    prototypes = np.zeros((10, 28, 28), dtype=np.float32)
    for label in range(10):
        row, column = 3 + (label // 5) * 14, 1 + (label % 5) * 5
        prototypes[label, row:row + 7, column:column + 4] = 200

    def make(count):
        labels = np.arange(count, dtype=np.int64) % 10
        rng.shuffle(labels)
        pixels = prototypes[labels] + rng.normal(15, 10, (count, 28, 28))
        return np.clip(pixels, 0, 255).astype(np.uint8), labels

    return (*make(600), *make(100))


def split_images(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    val_size: int = 10000,
    split_seed: int = 7,
    stats: tuple[float, float] | None = None,
) -> ImageSplits:
    """Reserve validation from official training data; normalize from train only."""
    for images, labels in ((train_images, train_labels), (test_images, test_labels)):
        if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != np.uint8:
            raise ValueError("Expected uint8 images with shape [N, 28, 28]")
        if labels.shape != (len(images),) or len(images) == 0:
            raise ValueError("Expected one label per image and nonempty partitions")
        if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= 10)):
            raise ValueError("Fashion-MNIST labels must be integers in [0, 9]")
    if not 0 < val_size < len(train_images):
        raise ValueError("val_size must leave nonempty training and validation splits")
    indices = np.random.default_rng(split_seed).permutation(len(train_images))
    val_indices, train_indices = indices[:val_size], indices[val_size:]

    def scale(images):
        return images[:, None].astype(np.float32) / 255.0

    train_x = scale(train_images[train_indices])
    val_x = scale(train_images[val_indices])
    test_x = scale(test_images)
    if stats is None:
        mean = float(train_x.mean(dtype=np.float64))
        std = max(float(train_x.std(dtype=np.float64)), 1e-7)
    else:
        mean, std = map(float, stats)
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError("Normalization statistics must be finite with positive std")
    for images in (train_x, val_x, test_x):
        images -= mean
        images /= std
    return ImageSplits(
        train_x, train_labels[train_indices].astype(np.int64),
        val_x, train_labels[val_indices].astype(np.int64),
        test_x, test_labels.astype(np.int64), mean, std,
    )


def get_image_splits(
    data_dir: Path = DEFAULT_DATA_DIR,
    demo: bool = False,
    val_size: int | None = None,
    split_seed: int = 7,
    stats: tuple[float, float] | None = None,
    download: bool = True,
) -> ImageSplits:
    arrays = demo_images() if demo else load_fashion_mnist(data_dir, download)
    if val_size is None:
        val_size = 100 if demo else 10000
    return split_images(*arrays, val_size=val_size, split_seed=split_seed, stats=stats)


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["fashion-mnist", "eeg"], default="fashion-mnist")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--demo", action="store_true", help="Synthetic data for offline checks")
    args = parser.parse_args(argv)
    if args.dataset == "eeg":
        splits = get_splits(args.subject, args.demo)
        print("SYNTHETIC EEG DEMO" if args.demo else "BCI Competition IV Dataset 2b (legacy)")
        print("Sessions:", splits.session_names)
        num_classes = 2
    else:
        splits = get_image_splits(
            args.data_dir, args.demo, args.val_size, args.split_seed,
            download=not args.no_download,
        )
        print("SYNTHETIC IMAGE DEMO" if args.demo else "Fashion-MNIST")
        print("Classes:", list(splits.class_names))
        num_classes = len(splits.class_names)
    for name in ("train", "val", "test"):
        x, y = getattr(splits, f"{name}_x"), getattr(splits, f"{name}_y")
        print(f"{name:5s}: {x.shape}, class counts={np.bincount(y, minlength=num_classes).tolist()}")


if __name__ == "__main__":
    main()
