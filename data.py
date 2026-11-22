"""Fashion-MNIST loading, stratified train/validation split and [0, 1] pixel scaling."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FASHION_CLASSES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
DEFAULT_DATA_DIR = Path("data/vision")
DEFAULT_VAL_SIZE = 6000
DEMO_VAL_SIZE = 100
DEFAULT_SPLIT_SEED = 7
# Saved with checkpoints so evaluation can refuse models built on other data handling.
SPLIT_METHOD = "stratified"
PREPROCESSING = "scale-0-1"


@dataclass
class ImageSplits:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_indices: np.ndarray  # positions in the official training partition
    val_indices: np.ndarray
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


def stratified_split_indices(labels: np.ndarray, val_size: int, split_seed: int):
    """Return sorted (train, validation) positions with class proportions preserved.

    Each class contributes val_size * class_share images to validation, rounded so
    the total is exactly val_size. The same split_seed always gives the same split.
    """
    if not 0 < val_size < len(labels):
        raise ValueError("val_size must leave nonempty training and validation splits")
    classes, counts = np.unique(labels, return_counts=True)
    exact = val_size * counts / len(labels)
    per_class = np.floor(exact).astype(np.int64)
    largest_remainders = np.argsort(-(exact - per_class), kind="stable")
    per_class[largest_remainders[:val_size - per_class.sum()]] += 1
    if np.any(per_class >= counts):
        raise ValueError("val_size must leave training images in every class")
    rng = np.random.default_rng(split_seed)
    val_indices = np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(labels == label))[:count]
        for label, count in zip(classes, per_class)
    ]))
    train_indices = np.setdiff1d(np.arange(len(labels)), val_indices)
    return train_indices, val_indices


def split_images(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    val_size: int = DEFAULT_VAL_SIZE,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> ImageSplits:
    """Reserve a stratified validation set from the official training data.

    Pixels are scaled to [0, 1] with no further normalization. The official
    test images are kept separate and only scaled.
    """
    for images, labels in ((train_images, train_labels), (test_images, test_labels)):
        if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != np.uint8:
            raise ValueError("Expected uint8 images with shape [N, 28, 28]")
        if labels.shape != (len(images),) or len(images) == 0:
            raise ValueError("Expected one label per image and nonempty partitions")
        if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= 10)):
            raise ValueError("Fashion-MNIST labels must be integers in [0, 9]")
    train_indices, val_indices = stratified_split_indices(train_labels, val_size, split_seed)
    if np.intersect1d(train_indices, val_indices).size or (
        len(train_indices) + len(val_indices) != len(train_labels)
    ):
        raise RuntimeError("Training and validation splits must be disjoint and complete")

    def scale(images):
        return images[:, None].astype(np.float32) / 255.0

    return ImageSplits(
        scale(train_images[train_indices]), train_labels[train_indices].astype(np.int64),
        scale(train_images[val_indices]), train_labels[val_indices].astype(np.int64),
        scale(test_images), test_labels.astype(np.int64),
        train_indices, val_indices,
    )


def get_image_splits(
    data_dir: Path = DEFAULT_DATA_DIR,
    demo: bool = False,
    val_size: int | None = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
    download: bool = True,
) -> ImageSplits:
    arrays = demo_images() if demo else load_fashion_mnist(data_dir, download)
    if val_size is None:
        val_size = DEMO_VAL_SIZE if demo else DEFAULT_VAL_SIZE
    return split_images(*arrays, val_size=val_size, split_seed=split_seed)


def split_summary(splits: ImageSplits, demo: bool, split_seed: int) -> dict:
    """Describe the splits. Test images are counted but their pixels are not inspected."""
    num_classes = len(splits.class_names)

    def describe(x, y, pixels=True):
        row = {"images": int(len(y)), "class_counts": np.bincount(y, minlength=num_classes).tolist()}
        if pixels:
            row.update({
                "pixel_mean": float(x.mean(dtype=np.float64)),
                "pixel_std": float(x.std(dtype=np.float64)),
                "pixel_min": float(x.min()), "pixel_max": float(x.max()),
            })
        return row

    return {
        "dataset": "synthetic-demo" if demo else "fashion-mnist",
        "split_method": SPLIT_METHOD,
        "split_seed": split_seed,
        "preprocessing": PREPROCESSING,
        "class_names": list(splits.class_names),
        "splits": {
            "train": describe(splits.train_x, splits.train_y),
            "val": describe(splits.val_x, splits.val_y),
            "test": describe(splits.test_x, splits.test_y, pixels=False),
        },
        "train_val_disjoint": not np.intersect1d(splits.train_indices, splits.val_indices).size,
        "val_indices_sha256": hashlib.sha256(splits.val_indices.astype("<i8").tobytes()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Synthetic data for offline checks")
    parser.add_argument("--summary-out", type=Path, help="Where to save the split summary JSON")
    args = parser.parse_args(argv)
    splits = get_image_splits(
        args.data_dir, args.demo, args.val_size, args.split_seed,
        download=not args.no_download,
    )
    summary = split_summary(splits, args.demo, args.split_seed)
    print("SYNTHETIC IMAGE DEMO" if args.demo else "Fashion-MNIST")
    print("Classes:", summary["class_names"])
    for name, row in summary["splits"].items():
        print(f"{name:5s}: {row['images']} images, class counts={row['class_counts']}")
    print("Train and validation disjoint:", summary["train_val_disjoint"])
    out = args.summary_out or Path(
        "results/data_summary_demo.json" if args.demo else "results/data_summary.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()
