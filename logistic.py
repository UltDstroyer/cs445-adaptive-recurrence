"""Sanity baseline: multinomial logistic regression on the 784 pixels.

Trained on the 54,000 training images with full-batch L-BFGS and a small fixed
L2 penalty (no tuning), then scored on the validation images only. The test
images are never used here.

Saves its validation outputs in the same format as evaluate.py, as a network
with a single exit, so sweep.py can compare it with the multi-exit network:
    results/logistic_val_outputs.npz and .json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import DEFAULT_DATA_DIR, DEFAULT_SPLIT_SEED, PREPROCESSING, SPLIT_METHOD, get_image_splits
from exit_outputs import exit_metrics, save_exit_outputs

DEFAULT_L2 = 1e-4
DEFAULT_MAX_ITERATIONS = 500


def fit_logistic_regression(x: np.ndarray, y: np.ndarray, num_classes: int,
                            l2: float = DEFAULT_L2, max_iterations: int = DEFAULT_MAX_ITERATIONS):
    """Weights [features, classes] and biases [classes] minimizing cross-entropy + l2 * |W|^2 / 2.

    Starts from zeros, so the result does not depend on a random seed.
    """
    features = torch.from_numpy(x.reshape(len(x), -1)).double()
    labels = torch.from_numpy(y)
    weights = torch.zeros(features.shape[1], num_classes, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(num_classes, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], lr=1, max_iter=max_iterations,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  history_size=20, line_search_fn="strong_wolfe")

    def objective():
        optimizer.zero_grad()
        loss = F.cross_entropy(features @ weights + bias, labels) + 0.5 * l2 * (weights ** 2).sum()
        loss.backward()
        return loss

    final_loss = optimizer.step(objective)
    iterations = optimizer.state[weights]["n_iter"]
    return weights.detach().numpy(), bias.detach().numpy(), float(final_loss.detach()), int(iterations)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument("--val-size", type=int)
    p.add_argument("--l2", type=float, default=DEFAULT_L2)
    p.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    p.add_argument("--out", type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--demo", action="store_true", help="Artificial images for offline checks")
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args(argv)
    if args.l2 < 0 or args.max_iterations < 1:
        p.error("l2 must be nonnegative and max-iterations positive")
    out = args.out or Path("results") / ("logistic_demo_val_outputs.npz" if args.demo else "logistic_val_outputs.npz")
    if out.suffix != ".npz":
        p.error("--out must end in .npz")
    if out.exists() and not args.overwrite:
        p.error(f"{out} already exists; add --overwrite to replace it")

    splits = get_image_splits(args.data_dir, args.demo, args.val_size, args.split_seed,
                              download=not args.no_download)
    num_classes = len(splits.class_names)
    start = time.perf_counter()
    weights, bias, train_loss, iterations = fit_logistic_regression(
        splits.train_x, splits.train_y, num_classes, args.l2, args.max_iterations)
    seconds = time.perf_counter() - start
    logits = splits.val_x.reshape(len(splits.val_x), -1).astype(np.float64) @ weights + bias
    features = weights.shape[0]
    metadata = {
        "split": "val",
        "demo": args.demo,
        "model": "logistic regression on pixels",
        "model_config": {"architecture": "logistic", "features": features, "num_classes": num_classes},
        "training_config": {"l2": args.l2, "optimizer": "full-batch L-BFGS from zeros",
                            "max_iterations": args.max_iterations, "iterations": iterations,
                            "final_training_objective": train_loss, "seconds": seconds},
        "data_config": {"dataset": "fashion-mnist", "demo": args.demo, "data_dir": str(args.data_dir.resolve()),
                        "val_size": len(splits.val_y), "split_seed": args.split_seed,
                        "split_method": SPLIT_METHOD, "preprocessing": PREPROCESSING},
        "class_names": list(splits.class_names),
        # One "exit": the whole model is a single linear layer.
        "multiply_adds": {"stem": 0, "blocks": [features * num_classes], "heads": [0]},
    }
    logits = logits[:, None, :].astype(np.float32)
    save_exit_outputs(out, logits, splits.val_y, splits.val_indices, metadata)
    metrics = exit_metrics(logits, splits.val_y)
    out.with_suffix(".json").write_text(json.dumps(
        {**metadata, "outputs": str(out), "images": len(splits.val_y),
         "macro_f1": metrics["exit_macro_f1"][0], "accuracy": metrics["exit_accuracy"][0],
         "loss": metrics["exit_loss"][0]}, indent=2), encoding="utf-8")
    print(("SYNTHETIC IMAGE DEMO, " if args.demo else "") + "logistic regression on pixels")
    print(f"{iterations} L-BFGS iterations in {seconds:.1f}s; validation macro-F1 "
          f"{metrics['exit_macro_f1'][0]:.3f}, accuracy {metrics['exit_accuracy'][0]:.3f}")
    print(f"Multiply-adds per image: {features * num_classes:,}")
    print("Saved:", out)


if __name__ == "__main__":
    main()
