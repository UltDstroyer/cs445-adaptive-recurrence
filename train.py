"""Train a feedforward DNN on Fashion-MNIST with validation checkpointing."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import DEFAULT_DATA_DIR, get_image_splits
from model import DEFAULT_DROPOUT, DEFAULT_HIDDEN_SIZES, DNNClassifier


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def make_loader(x, y, batch_size, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.inference_mode()
def evaluate_split(model, x, y, device, batch_size=128):
    model.eval()
    loss, correct = 0.0, 0
    for images, labels in make_loader(x, y, batch_size):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += int((logits.argmax(-1) == labels).sum())
    return {"loss": loss / len(y), "accuracy": correct / len(y)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=list(DEFAULT_HIDDEN_SIZES))
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--patience", type=int, default=0, help="Stop after this many unimproved epochs; 0 disables")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument("--val-size", type=int)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out", type=Path, default=Path("checkpoints/dnn.pt"))
    p.add_argument("--history", type=Path)
    p.add_argument("--demo", action="store_true", help="Artificial images for offline checks")
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args(argv)
    if min(args.epochs, args.batch_size, *args.hidden_sizes) < 1:
        p.error("epochs, batch-size and hidden-sizes must be positive")
    if not 0 <= args.dropout < 1 or args.patience < 0:
        p.error("dropout must be in [0, 1); patience cannot be negative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error("lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("weight-decay must be finite and nonnegative")
    if args.val_size is not None and args.val_size < 1:
        p.error("val-size must be positive")
    if args.split_seed < 0:
        p.error("split-seed must be nonnegative")
    history_path = args.history or args.out.with_name(f"{args.out.stem}_training.json")
    if history_path.resolve() == args.out.resolve():
        p.error("history and checkpoint paths must differ")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    splits = get_image_splits(
        args.data_dir, args.demo, args.val_size, args.split_seed,
        download=not args.no_download,
    )
    device = select_device(args.device)
    config = {
        "input_shape": list(splits.train_x.shape[1:]),
        "hidden_sizes": list(args.hidden_sizes),
        "num_classes": len(splits.class_names),
        "dropout": args.dropout,
    }
    data_config = {
        "dataset": "fashion-mnist",
        "demo": args.demo,
        "data_dir": str(args.data_dir.resolve()),
        "val_size": len(splits.val_y),
        "split_seed": args.split_seed,
    }
    training_config = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "patience": args.patience,
        "seed": args.seed, "device": str(device),
    }
    model = DNNClassifier(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(splits.train_x, splits.train_y, args.batch_size, shuffle=True)
    print("SYNTHETIC IMAGE DEMO" if args.demo else "Fashion-MNIST", flush=True)
    print("Feedforward DNN:", config, flush=True)
    print(f"Examples: train={len(splits.train_y)}, val={len(splits.val_y)}, test={len(splits.test_y)}", flush=True)
    best_key = (-1.0, -float("inf"))
    best_epoch, stale_epochs = 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
            correct += int((logits.detach().argmax(-1) == labels).sum())
        val = evaluate_split(model, splits.val_x, splits.val_y, device, args.batch_size)
        row = {
            "epoch": epoch, "train_loss": total_loss / len(splits.train_y),
            "train_accuracy": correct / len(splits.train_y),
            "val_loss": val["loss"], "val_accuracy": val["accuracy"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: train loss={row['train_loss']:.4f}, "
            f"train accuracy={row['train_accuracy']:.3f}, "
            f"val loss={row['val_loss']:.4f}, val accuracy={row['val_accuracy']:.3f}",
            flush=True,
        )
        key = (val["accuracy"], -val["loss"])
        if key > best_key:
            best_key, best_epoch, stale_epochs = key, epoch, 0
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_type": "dnn", "model_state": model.state_dict(),
                "model_config": config, "data_config": data_config,
                "training_config": training_config,
                "class_names": list(splits.class_names),
                "pixel_mean": splits.mean, "pixel_std": splits.std,
                "val_accuracy": val["accuracy"], "val_loss": val["loss"], "epoch": epoch,
            }, args.out)
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            break

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "model_type": "dnn", "model_config": config, "data_config": data_config,
        "training_config": training_config, "history": history,
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "stopped_early": len(history) < args.epochs,
    }, indent=2), encoding="utf-8")
    print(f"Best validation accuracy: {best_key[0]:.3f}")
    print("Checkpoint:", args.out)
    print("Training history:", history_path)
    print("Next: python evaluate.py --checkpoint", args.out)
    if args.demo:
        print("Synthetic demo accuracy is not a Fashion-MNIST benchmark result.")


if __name__ == "__main__":
    main()
