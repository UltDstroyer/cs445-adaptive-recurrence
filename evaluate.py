"""Compare fixed and adaptive inference on held-out EEG sessions."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import get_splits
from model import AdaptiveRecurrentClassifier
from train import select_device


def accuracy(logits: torch.Tensor, labels: np.ndarray) -> float:
    return float((logits.argmax(dim=-1).cpu().numpy() == labels).mean())


def measure_time(model, x, batch_size, depth, confidence, stability, min_steps, repeats=3):
    """Full inference time, including encoder and stopping overhead."""
    batches = list(x.split(batch_size))

    def run():
        for batch in batches:
            if depth is None:
                model.predict_adaptive(batch, confidence, stability, min_steps)
            else:
                model.forward_fixed(batch, depth)
        if x.device.type == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        run()
        start = time.perf_counter()
        for _ in range(repeats):
            run()
    return (time.perf_counter() - start) * 1000 / (repeats * len(x))


def adaptive_predictions(model, x, batch_size, confidence, stability, min_steps):
    logits, steps = [], []
    with torch.inference_mode():
        for batch in x.split(batch_size):
            pred, used = model.predict_adaptive(batch, confidence, stability, min_steps)
            logits.append(pred.cpu())
            steps.append(used.cpu())
    return torch.cat(logits), torch.cat(steps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    p.add_argument("--confidence", type=float, default=0.85)
    p.add_argument("--stability", type=float, default=0.05)
    p.add_argument("--min-steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out", type=Path, default=Path("results/evaluation.json"))
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    demo = checkpoint["demo"]
    subject = checkpoint["subject"]
    stats = (checkpoint["channel_mean"], checkpoint["channel_std"])
    splits = get_splits(subject, demo=demo, stats=stats)
    if splits.session_names != checkpoint["sessions"]:
        raise ValueError("Dataset session list differs from training")

    device = select_device(args.device)
    model = AdaptiveRecurrentClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    results = {
        "subject": subject,
        "demo": demo,
        "checkpoint_epoch": checkpoint["epoch"],
        "sessions": splits.session_names,
        "confidence": args.confidence,
        "stability": args.stability,
        "min_steps": args.min_steps,
        "splits": {},
    }
    print("SYNTHETIC DEMO DATA: NOT EEG RESEARCH RESULTS" if demo else "EEG classification evaluation")
    for split_name in ("val", "test"):
        x_np = getattr(splits, f"{split_name}_x")
        labels = getattr(splits, f"{split_name}_y")
        x = torch.from_numpy(x_np).to(device)
        with torch.inference_mode():
            all_logits = model(x).cpu()
        fixed_accuracies = [accuracy(all_logits[:, i], labels) for i in range(model.max_steps)]
        adaptive_logits, used = adaptive_predictions(
            model, x, args.batch_size, args.confidence, args.stability, args.min_steps
        )
        adaptive_accuracy = accuracy(adaptive_logits, labels)
        fixed_ms = measure_time(model, x, args.batch_size, model.max_steps,
                                args.confidence, args.stability, args.min_steps)
        adaptive_ms = measure_time(model, x, args.batch_size, None,
                                   args.confidence, args.stability, args.min_steps)
        results["splits"][split_name] = {
            "examples": len(labels),
            "fixed_accuracy_by_step": fixed_accuracies,
            "adaptive_accuracy": adaptive_accuracy,
            "average_steps": float(used.float().mean()),
            "exit_counts": {str(n): int((used == n).sum()) for n in range(1, model.max_steps + 1)},
            "fixed_ms_per_trial": fixed_ms,
            "adaptive_ms_per_trial": adaptive_ms,
        }
        print(f"\n{split_name.upper()} ({len(labels)} trials)")
        print("  Fixed accuracy by step:", [round(v, 3) for v in fixed_accuracies])
        print(f"  Adaptive accuracy: {adaptive_accuracy:.3f}")
        print(f"  Mean steps: {used.float().mean():.2f} / {model.max_steps}")
        print(f"  Actual latency: fixed {fixed_ms:.3f}, adaptive {adaptive_ms:.3f} ms/trial")
        print("  Exit counts:", results["splits"][split_name]["exit_counts"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved:", args.out)
    print("Thresholds are exploratory, not optimized on the held-out test set.")
    if demo:
        print("Synthetic-demo accuracy is NOT BCI performance.")


if __name__ == "__main__":
    main()
