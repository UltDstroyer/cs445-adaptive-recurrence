"""Train a compact EEG classifier with supervised predictions at every pass."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import get_splits
from model import AdaptiveRecurrentClassifier


def select_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def predict_accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(x).to(device))
        predicted = logits[:, -1].argmax(dim=-1).cpu().numpy()
    return float((predicted == y).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out", type=Path, default=Path("checkpoints/model.pt"))
    p.add_argument("--demo", action="store_true", help="Synthetic data for pipeline checks only")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_size) < 1 or args.steps < 2:
        p.error("epochs, batch-size, hidden-size must be positive; steps must be at least 2")
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
    }
    model = AdaptiveRecurrentClassifier(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_data = TensorDataset(torch.from_numpy(splits.train_x), torch.from_numpy(splits.train_y))
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits_by_step = model(x)
            # Equally weight all depths so early predictions can become useful.
            loss = sum(
                F.cross_entropy(logits_by_step[:, step], y)
                for step in range(model.max_steps)
            ) / model.max_steps
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        val_accuracy = predict_accuracy(model, splits.val_x, splits.val_y, device)
        print(
            f"Epoch {epoch:02d}: loss={total_loss / len(train_data):.4f}, "
            f"val accuracy (full)={val_accuracy:.3f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "model_config": config,
                "subject": args.subject,
                "demo": args.demo,
                "seed": args.seed,
                "sessions": splits.session_names,
                "channel_mean": splits.mean.squeeze().tolist(),
                "channel_std": splits.std.squeeze().tolist(),
                "val_accuracy": val_accuracy,
                "epoch": epoch,
            }, args.out)

    print(f"Best validation accuracy: {best_accuracy:.3f}")
    print(f"Checkpoint: {args.out}")
    print("Next: python evaluate.py --checkpoint", args.out)
    if args.demo:
        print("WARNING: synthetic demo data, not EEG results.")


if __name__ == "__main__":
    main()
