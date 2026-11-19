"""Train an EEG encoder and shared recurrence with validation-based stopping."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import get_splits
from model import (
    DEFAULT_DROPOUT,
    DEFAULT_ENCODER,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MAX_STEPS,
    AdaptiveRecurrentClassifier,
)


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def evaluate_split(model, x, y, device, batch_size=32):
    """Full-depth loss and accuracy with dropout disabled and batch norm frozen."""
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(y), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            labels = torch.from_numpy(y[start:start + batch_size]).to(device)
            logits = model.forward_fixed(batch)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
    return {"loss": total_loss / len(y), "accuracy": correct / len(y)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    p.add_argument("--patience", type=int, default=15, help="Epochs without checkpoint improvement; 0 disables stopping")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    p.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--encoder", choices=["eeg", "dense"], default=DEFAULT_ENCODER)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out", type=Path, default=Path("checkpoints/eeg_model.pt"))
    p.add_argument("--history", type=Path, help="JSON history; defaults to results/<checkpoint stem>_training.json")
    p.add_argument("--demo", action="store_true", help="Synthetic data for pipeline checks only")
    args = p.parse_args(argv)
    if min(args.epochs, args.batch_size, args.hidden_size) < 1 or args.steps < 2:
        p.error("epochs, batch-size, hidden-size must be positive; steps must be at least 2")
    if args.patience < 0 or not 0 <= args.dropout < 1:
        p.error("patience must be nonnegative; dropout must be in [0, 1)")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("lr must be positive and weight-decay nonnegative, both finite")
    if args.history is None:
        args.history = Path("results") / f"{args.out.stem}_training.json"
    if args.history.resolve() == args.out.resolve():
        p.error("history and checkpoint must use different paths")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    splits = get_splits(args.subject, args.demo)
    device = select_device(args.device)
    config = {
        "channels": int(splits.train_x.shape[1]),
        "samples": int(splits.train_x.shape[2]),
        "hidden_size": args.hidden_size,
        "num_classes": 2,
        "max_steps": args.steps,
        "encoder_type": args.encoder,
        "dropout": args.dropout,
    }
    training_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    print("Model:", json.dumps(config, sort_keys=True), flush=True)
    print(f"Training: subject={args.subject}, seed={args.seed}, lr={args.lr}, device={device}, max epochs={args.epochs}, patience={args.patience}", flush=True)
    model = AdaptiveRecurrentClassifier(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_data = TensorDataset(torch.from_numpy(splits.train_x), torch.from_numpy(splits.train_y))
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    best_accuracy, best_val_loss = -1.0, float("inf")
    best_epoch, stale_epochs = 0, 0
    history = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            # Equal supervision at every recurrent depth, as in the original model.
            targets = y[:, None].expand(-1, model.max_steps).reshape(-1)
            loss = F.cross_entropy(logits.reshape(-1, model.num_classes), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        train_metrics = evaluate_split(model, splits.train_x, splits.train_y, device, args.batch_size)
        val_metrics = evaluate_split(model, splits.val_x, splits.val_y, device, args.batch_size)
        if not all(math.isfinite(value) for value in (*train_metrics.values(), *val_metrics.values())):
            raise RuntimeError(f"Non-finite evaluation metric at epoch {epoch}")
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_data),
            "train_eval_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: train loss={row['train_loss']:.4f}, "
            f"train accuracy={row['train_accuracy']:.3f}, "
            f"val loss={row['val_loss']:.4f}, val accuracy={row['val_accuracy']:.3f}",
            flush=True,
        )
        # Keep the original criterion, using loss to resolve accuracy ties.
        improved = row["val_accuracy"] > best_accuracy or (
            row["val_accuracy"] == best_accuracy and row["val_loss"] < best_val_loss
        )
        if improved:
            best_accuracy, best_val_loss = row["val_accuracy"], row["val_loss"]
            best_epoch, stale_epochs = epoch, 0
            torch.save({
                "schema_version": 2,
                "model_state": model.state_dict(),
                "model_config": config,
                "training_config": training_config,
                "subject": args.subject,
                "demo": args.demo,
                "seed": args.seed,
                "sessions": splits.session_names,
                "channel_mean": splits.mean.squeeze().tolist(),
                "channel_std": splits.std.squeeze().tolist(),
                "val_accuracy": best_accuracy,
                "val_loss": best_val_loss,
                "epoch": epoch,
            }, args.out)
        else:
            stale_epochs += 1
        summary = {
            "subject": args.subject,
            "demo": args.demo,
            "seed": args.seed,
            "device": str(device),
            "model_config": config,
            "training_config": training_config,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "sessions": splits.session_names,
            "split_examples": {name: len(getattr(splits, f"{name}_y")) for name in ("train", "val", "test")},
            "selection": "maximum full-depth validation accuracy; lower validation loss breaks ties",
            "best_epoch": best_epoch,
            "best_val_accuracy": best_accuracy,
            "best_val_loss": best_val_loss,
            "epochs_completed": epoch,
            "stopped_early": bool(args.patience and stale_epochs >= args.patience and epoch < args.epochs),
            "history": history,
        }
        args.history.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if args.patience and stale_epochs >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement.", flush=True)
            break

    print(f"Best validation accuracy: {best_accuracy:.4f} at epoch {best_epoch}", flush=True)
    print(f"Checkpoint: {args.out}", flush=True)
    print(f"History: {args.history}", flush=True)
    if args.demo:
        print("WARNING: synthetic demo data, not EEG results.", flush=True)


if __name__ == "__main__":
    main()
