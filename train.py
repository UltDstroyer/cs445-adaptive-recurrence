"""Train a multi-exit network (CNN by default) on Fashion-MNIST with a loss at every exit.

The training loss is the cross-entropy averaged over all exits. After each epoch
every exit is scored on the validation images, and the checkpoint keeps the
epoch with the best macro-F1 averaged across exits (ties go to lower loss).
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import DEFAULT_DATA_DIR, DEFAULT_SPLIT_SEED, PREPROCESSING, SPLIT_METHOD, get_image_splits
from model import (
    ARCHITECTURES, DEFAULT_BLOCK_HIDDEN, DEFAULT_CNN_CHANNELS, DEFAULT_CNN_DOWNSAMPLE_AT,
    DEFAULT_NUM_EXITS, DEFAULT_WIDTH, build_model,
)

PLANNED_SEEDS = 5


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def make_loader(x, y, batch_size, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def multi_exit_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged over every exit and image; logits are [batch, exits, classes]."""
    return F.cross_entropy(logits.flatten(0, 1), labels.repeat_interleave(logits.shape[1]))


def macro_f1(labels: np.ndarray, predicted: np.ndarray, num_classes: int) -> float:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(
        2 * confusion.diagonal(), denominator,
        out=np.zeros(num_classes, dtype=np.float64), where=denominator > 0,
    )
    return float(f1.mean())


@torch.inference_mode()
def evaluate_split(model, x, y, device, batch_size=128):
    """Loss, accuracy and macro-F1 for every exit on one split."""
    model.eval()
    loss_sums = torch.zeros(model.num_exits, dtype=torch.float64)
    predictions = []
    for images, labels in make_loader(x, y, batch_size):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        per_image = F.cross_entropy(logits.transpose(1, 2), labels[:, None].expand(-1, model.num_exits), reduction="none")
        loss_sums += per_image.sum(dim=0).double().cpu()
        predictions.append(logits.argmax(-1).cpu())
    predicted = torch.cat(predictions).numpy()  # [images, exits]
    exit_loss = (loss_sums / len(y)).tolist()
    exit_accuracy = [float((predicted[:, k] == y).mean()) for k in range(model.num_exits)]
    exit_macro_f1 = [macro_f1(y, predicted[:, k], model.num_classes) for k in range(model.num_exits)]
    return {
        "loss": float(np.mean(exit_loss)),
        "mean_macro_f1": float(np.mean(exit_macro_f1)),
        "exit_loss": exit_loss,
        "exit_accuracy": exit_accuracy,
        "exit_macro_f1": exit_macro_f1,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="cnn",
                   help="cnn (default) or mlp (the earlier design, which failed the exit-gap check)")
    p.add_argument("--channels", type=int, nargs=2, default=list(DEFAULT_CNN_CHANNELS),
                   help="CNN: channels before and after the downsampling block")
    p.add_argument("--downsample-at", type=int, default=DEFAULT_CNN_DOWNSAMPLE_AT,
                   help="CNN: block that halves the resolution")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="MLP only")
    p.add_argument("--block-hidden", type=int, default=DEFAULT_BLOCK_HIDDEN, help="MLP only")
    p.add_argument("--num-exits", type=int, default=DEFAULT_NUM_EXITS)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--patience", type=int, default=0, help="Stop after this many unimproved epochs; 0 disables")
    p.add_argument("--seed", type=int, default=0, help="Model initialization and training randomness")
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED,
                   help="Keep fixed across training seeds so every run uses the same validation images")
    p.add_argument("--val-size", type=int)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out", type=Path, default=Path("checkpoints/multi_exit.pt"))
    p.add_argument("--history", type=Path)
    p.add_argument("--demo", action="store_true", help="Artificial images for offline checks")
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args(argv)
    if min(args.epochs, args.batch_size, args.width, args.block_hidden, args.num_exits, *args.channels) < 1:
        p.error("epochs, batch-size, channels, width, block-hidden and num-exits must be positive")
    if args.architecture == "cnn" and not 2 <= args.downsample_at <= args.num_exits:
        p.error("downsample-at must be between 2 and num-exits")
    if args.patience < 0:
        p.error("patience cannot be negative")
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
        "architecture": args.architecture,
        "input_shape": list(splits.train_x.shape[1:]),
        "num_exits": args.num_exits,
        "num_classes": len(splits.class_names),
    }
    if args.architecture == "cnn":
        config.update(channels=list(args.channels), downsample_at=args.downsample_at)
    else:
        config.update(width=args.width, block_hidden=args.block_hidden)
    data_config = {
        "dataset": "fashion-mnist",
        "demo": args.demo,
        "data_dir": str(args.data_dir.resolve()),
        "val_size": len(splits.val_y),
        "split_seed": args.split_seed,
        "split_method": SPLIT_METHOD,
        "preprocessing": PREPROCESSING,
    }
    training_config = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "patience": args.patience,
        "seed": args.seed, "device": str(device), "loss": "mean cross-entropy over exits",
        "selection": "best mean validation macro-F1 across exits, ties to lower loss",
    }
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(splits.train_x, splits.train_y, args.batch_size, shuffle=True)
    print("SYNTHETIC IMAGE DEMO" if args.demo else "Fashion-MNIST", flush=True)
    print(f"Multi-exit {args.architecture.upper()}:", config,
          f"({sum(p.numel() for p in model.parameters()):,} parameters)", flush=True)
    print(f"Examples: train={len(splits.train_y)}, val={len(splits.val_y)}", flush=True)
    best_key = (-1.0, -float("inf"))
    best_epoch, stale_epochs = 0, 0
    history = []
    run_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = multi_exit_loss(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        val = evaluate_split(model, splits.val_x, splits.val_y, device, args.batch_size)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(splits.train_y),
            "val_loss": val["loss"],
            "val_mean_macro_f1": val["mean_macro_f1"],
            "val_exit_macro_f1": val["exit_macro_f1"],
            "val_exit_accuracy": val["exit_accuracy"],
            "val_exit_loss": val["exit_loss"],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} ({row['seconds']:.1f}s): train loss={row['train_loss']:.4f}, "
            f"val loss={row['val_loss']:.4f}, val macro-F1 mean={row['val_mean_macro_f1']:.3f}, "
            f"exit 1={val['exit_macro_f1'][0]:.3f}, last exit={val['exit_macro_f1'][-1]:.3f}",
            flush=True,
        )
        key = (val["mean_macro_f1"], -val["loss"])
        if key > best_key:
            best_key, best_epoch, stale_epochs = key, epoch, 0
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_type": "multi_exit", "model_state": model.state_dict(),
                "model_config": config, "data_config": data_config,
                "training_config": training_config,
                "class_names": list(splits.class_names),
                "val_mean_macro_f1": val["mean_macro_f1"], "val_loss": val["loss"],
                "val_exit_macro_f1": val["exit_macro_f1"], "val_exit_accuracy": val["exit_accuracy"],
                "epoch": epoch,
            }, args.out)
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            break
    training_seconds = time.perf_counter() - run_start

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "model_type": "multi_exit", "model_config": config, "data_config": data_config,
        "training_config": training_config, "history": history,
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "stopped_early": len(history) < args.epochs,
        "training_seconds": training_seconds,
    }, indent=2), encoding="utf-8")
    best = history[best_epoch - 1]
    print(f"Best epoch {best_epoch}: mean validation macro-F1 {best['val_mean_macro_f1']:.3f}")
    print("Validation macro-F1 by exit:", [round(v, 3) for v in best["val_exit_macro_f1"]])
    print(f"Training time: {training_seconds / 60:.1f} min on {device}; "
          f"about {PLANNED_SEEDS * training_seconds / 60:.0f} min for {PLANNED_SEEDS} seeds")
    print("Checkpoint:", args.out)
    print("Training history:", history_path)
    if args.demo:
        print("Synthetic demo results are not Fashion-MNIST benchmark results.")


if __name__ == "__main__":
    main()
