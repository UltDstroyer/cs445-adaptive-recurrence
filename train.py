"""Train a multi-exit network (CNN by default) on Fashion-MNIST with a loss at every exit.

The training loss is the cross-entropy averaged over all exits. After each epoch
every exit is scored on the validation images, and the checkpoint keeps the
epoch with the best macro-F1 averaged across exits (ties go to lower loss).

Two ways to train:
  * Fast, on a GPU: --device auto (the default) uses an NVIDIA GPU (cuda) or an
    Apple-silicon GPU (mps) when one is available.
  * Slow computer, on the CPU: --device cpu, optionally --threads N to leave cores
    free. Progress is saved after every epoch, so a run can be stopped (Ctrl+C or
    closing the laptop) and continued later with --resume.
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

DEFAULT_EPOCHS = 30
PLANNED_SEEDS = 5
DEVICE_CHOICES = ["auto", "cpu", "cuda", "mps"]


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def select_device(choice: str) -> torch.device:
    """auto prefers an NVIDIA GPU, then an Apple-silicon GPU, then the CPU."""
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if mps_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested, but no NVIDIA GPU is available")
    if choice == "mps" and not mps_available():
        raise ValueError("--device mps was requested, but no Apple-silicon GPU is available")
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
        # Move to the CPU before converting: Apple GPUs do not support float64.
        loss_sums += per_image.sum(dim=0).cpu().double()
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


def resume_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_resume.pt")


def save_atomically(obj, path: Path) -> None:
    """Write to a temporary file first so an interruption never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
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
    p.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                   help="auto uses an NVIDIA (cuda) or Apple-silicon (mps) GPU when available, else the CPU")
    p.add_argument("--threads", type=int, help="CPU threads to use; lower it to keep a slow computer usable")
    p.add_argument("--resume", action="store_true",
                   help="Continue an interrupted run from its last finished epoch")
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
    if args.threads is not None and args.threads < 1:
        p.error("threads must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error("lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("weight-decay must be finite and nonnegative")
    if args.val_size is not None and args.val_size < 1:
        p.error("val-size must be positive")
    if args.split_seed < 0:
        p.error("split-seed must be nonnegative")
    history_path = args.history or args.out.with_name(f"{args.out.stem}_training.json")
    state_path = resume_path(args.out)
    if len({history_path.resolve(), args.out.resolve(), state_path.resolve()}) < 3:
        p.error("history and checkpoint paths must differ")
    if args.resume and not state_path.exists():
        p.error(f"--resume: no saved progress at {state_path}; run without --resume to start")
    try:
        device = select_device(args.device)
    except ValueError as error:
        p.error(str(error))
    if args.threads:
        torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    splits = get_image_splits(
        args.data_dir, args.demo, args.val_size, args.split_seed,
        download=not args.no_download,
    )
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
    # Everything that must stay the same for a resumed run to continue the same experiment.
    run_settings = {
        "model_config": config,
        "data": {k: v for k, v in data_config.items() if k != "data_dir"},
        "training": {"batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
                     "patience": args.patience, "seed": args.seed},
    }
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history, devices = [], []
    best_key, best_epoch, stale_epochs, earlier_seconds = (-1.0, -float("inf")), 0, 0, 0.0
    start_epoch, finished = 1, False
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["run_settings"] != run_settings:
            p.error(f"--resume: the saved progress at {state_path} used different settings; "
                    "use the same options as the original run (only --epochs, --device and --threads may change)")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        torch.set_rng_state(state["torch_rng_state"])
        history, devices = state["history"], state["devices"]
        best_key, best_epoch = tuple(state["best_key"]), state["best_epoch"]
        stale_epochs, earlier_seconds = state["stale_epochs"], state["training_seconds"]
        start_epoch, finished = state["epoch"] + 1, state["stopped_early"]
        print(f"Resuming after epoch {state['epoch']} (best so far: epoch {best_epoch})", flush=True)
    elif state_path.exists():
        print(f"Starting over; earlier saved progress at {state_path} will be replaced", flush=True)
    if not devices or devices[-1] != str(device):
        devices.append(str(device))

    training_config = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "patience": args.patience,
        "seed": args.seed, "device": str(device), "devices": devices,
        "loss": "mean cross-entropy over exits",
        "selection": "best mean validation macro-F1 across exits, ties to lower loss",
    }
    loader = make_loader(splits.train_x, splits.train_y, args.batch_size, shuffle=True)
    print("SYNTHETIC IMAGE DEMO" if args.demo else "Fashion-MNIST", flush=True)
    print(f"Multi-exit {args.architecture.upper()}:", config,
          f"({sum(p.numel() for p in model.parameters()):,} parameters)", flush=True)
    print(f"Examples: train={len(splits.train_y)}, val={len(splits.val_y)}; device={device}; "
          f"CPU threads={torch.get_num_threads()}", flush=True)

    session_start = time.perf_counter()
    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if finished:
                break
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
                "device": str(device),
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
                save_atomically({
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
            finished = bool(args.patience and stale_epochs >= args.patience)
            save_atomically({
                "run_settings": run_settings, "epoch": epoch,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(), "history": history, "devices": devices,
                "best_key": list(best_key), "best_epoch": best_epoch, "stale_epochs": stale_epochs,
                "training_seconds": earlier_seconds + time.perf_counter() - session_start,
                "stopped_early": finished,
            }, state_path)
    except KeyboardInterrupt:
        print(f"\nInterrupted during epoch {epoch}. Progress through the last completed epoch is "
              f"saved in {state_path}; rerun the same command with --resume to continue.", flush=True)
        raise SystemExit(130)
    training_seconds = earlier_seconds + time.perf_counter() - session_start

    if not history:
        p.error("no epochs were trained")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "model_type": "multi_exit", "model_config": config, "data_config": data_config,
        "training_config": training_config, "history": history,
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "stopped_early": len(history) < args.epochs,
        "training_seconds": training_seconds,
    }, indent=2), encoding="utf-8")
    best = history[best_epoch - 1]
    epoch_seconds = float(np.mean([row["seconds"] for row in history]))
    print(f"Best epoch {best_epoch}: mean validation macro-F1 {best['val_mean_macro_f1']:.3f}")
    print("Validation macro-F1 by exit:", [round(v, 3) for v in best["val_exit_macro_f1"]])
    print(f"Training time: {training_seconds / 60:.1f} min on {', '.join(devices)}; at this speed "
          f"{PLANNED_SEEDS} seeds x {args.epochs} epochs take about "
          f"{PLANNED_SEEDS * args.epochs * epoch_seconds / 60:.0f} min")
    print("Checkpoint:", args.out)
    print("Training history:", history_path)
    if args.demo:
        print("Synthetic demo results are not Fashion-MNIST benchmark results.")


if __name__ == "__main__":
    main()
