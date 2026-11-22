"""Evaluate a feedforward DNN on held-out Fashion-MNIST images."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import PREPROCESSING, SPLIT_METHOD, get_image_splits
from model import DNNClassifier
from train import DEVICE_CHOICES, make_loader, select_device


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work so timings measure the computation itself."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def evaluate_images(model, x, y, device, batch_size=128):
    model.eval()
    confusion = np.zeros((model.num_classes, model.num_classes), dtype=np.int64)
    total_loss = 0.0
    for images, labels in make_loader(x, y, batch_size):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        predicted = logits.argmax(-1).cpu().numpy()
        np.add.at(confusion, (labels.cpu().numpy(), predicted), 1)
    support = confusion.sum(axis=1)
    f1_denominator = support + confusion.sum(axis=0)
    class_f1 = np.divide(
        2 * confusion.diagonal(), f1_denominator,
        out=np.zeros(model.num_classes, dtype=np.float64), where=f1_denominator > 0,
    )
    return {
        "examples": len(y), "loss": total_loss / len(y),
        "accuracy": float(confusion.trace() / len(y)),
        "macro_f1": float(class_f1.mean()),
        "class_counts": support.tolist(), "confusion_matrix": confusion.tolist(),
    }


@torch.inference_mode()
def measure_latency(model, x, device, batch_size, repeats=3):
    """Warm up and time batched forward passes, excluding data loading/transfer."""
    images = torch.from_numpy(x[:batch_size]).to(device)
    model.eval()
    for _ in range(3):
        model(images)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model(images)
    synchronize(device)
    return (time.perf_counter() - start) * 1000 / (repeats * len(images))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/dnn.pt"))
    p.add_argument("--data-dir", type=Path, help="Override the dataset location saved in the checkpoint")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    p.add_argument("--out", type=Path, default=Path("results/dnn_evaluation.json"))
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    if args.out.resolve() == args.checkpoint.resolve():
        p.error("evaluation output and checkpoint paths must differ")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") == "multi_exit":
        p.error("evaluate.py does not support multi-exit checkpoints yet; "
                "separate validation and test evaluation is still being built")
    if checkpoint.get("model_type") != "dnn":
        p.error("Expected a DNN checkpoint produced by train.py; this is a legacy or unsupported model")
    data_config = checkpoint["data_config"]
    if data_config["dataset"] != "fashion-mnist":
        p.error("Unsupported checkpoint dataset")
    if (data_config.get("split_method"), data_config.get("preprocessing")) != (SPLIT_METHOD, PREPROCESSING):
        p.error("Checkpoint was trained with an older data split or pixel normalization; retrain it with train.py")
    data_dir = args.data_dir or Path(data_config["data_dir"])
    splits = get_image_splits(
        data_dir=data_dir, demo=data_config["demo"],
        val_size=data_config["val_size"], split_seed=data_config["split_seed"],
        download=not args.no_download,
    )
    if list(splits.class_names) != checkpoint["class_names"]:
        raise ValueError("Dataset class names differ from training")
    device = select_device(args.device)
    model = DNNClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    results = {
        "model_type": "dnn", "model_config": checkpoint["model_config"],
        "dataset": data_config["dataset"], "demo": data_config["demo"],
        "data_config": {**data_config, "data_dir": str(data_dir.resolve())},
        "class_names": list(splits.class_names), "checkpoint_epoch": checkpoint["epoch"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device), "splits": {},
    }
    print("SYNTHETIC IMAGE DEMO" if results["demo"] else "Fashion-MNIST DNN evaluation")
    for name in ("val", "test"):
        x, y = getattr(splits, f"{name}_x"), getattr(splits, f"{name}_y")
        metrics = evaluate_images(model, x, y, device, args.batch_size)
        metrics["forward_ms_per_image"] = measure_latency(model, x, device, args.batch_size)
        metrics["latency_batch_size"] = min(args.batch_size, len(y))
        results["splits"][name] = metrics
        print(f"{name.upper()}: accuracy={metrics['accuracy']:.3f}, macro-F1={metrics['macro_f1']:.3f}, loss={metrics['loss']:.4f}")
    results["latency_protocol"] = "Three warm-ups and three repeats on the first batch; forward only; excludes loading and device transfers."
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved:", args.out)
    if results["demo"]:
        print("Synthetic demo accuracy is not a Fashion-MNIST benchmark result.")


if __name__ == "__main__":
    main()
