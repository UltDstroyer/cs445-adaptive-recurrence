"""Run a trained multi-exit network over one split and save every exit's outputs.

Validation images are the default. The official test images are reserved for the
single final evaluation: scoring them needs --split test --confirm-final-test,
and an existing test output file is never overwritten.

Each run writes two files next to each other:
    <name>.npz   every exit's logits, the labels and the image indices
                 (read them with exit_outputs.load_exit_outputs; no PyTorch needed)
    <name>.json  per-exit accuracy, macro-F1 and loss, plus the run's settings
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data import PREPROCESSING, SPLIT_METHOD, get_image_splits
from exit_outputs import exit_metrics, save_exit_outputs
from model import build_model
from train import DEVICE_CHOICES, make_loader, select_device


@torch.inference_mode()
def collect_logits(model, x, device, batch_size=256) -> np.ndarray:
    """Every exit's logits for every image: [images, exits, classes] on the CPU."""
    model.eval()
    labels = np.zeros(len(x), dtype=np.int64)  # make_loader needs labels; they are not used here
    batches = [model(images.to(device)).float().cpu() for images, _ in make_loader(x, labels, batch_size)]
    return torch.cat(batches).numpy()


def default_output(checkpoint: Path, split: str) -> Path:
    return Path("results") / f"{checkpoint.stem}_{split}_outputs.npz"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/multi_exit.pt"))
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--confirm-final-test", action="store_true",
                   help="Required with --split test: the test set is scored once, at the end")
    p.add_argument("--out", type=Path, help="Output .npz path (default: results/<checkpoint>_<split>_outputs.npz)")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing validation output")
    p.add_argument("--data-dir", type=Path, help="Override the dataset location saved in the checkpoint")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    if args.split == "test" and not args.confirm_final_test:
        p.error("the test split is reserved for the single final evaluation; "
                "add --confirm-final-test only when running it")
    out = args.out or default_output(args.checkpoint, args.split)
    if out.suffix != ".npz":
        p.error("--out must end in .npz")
    summary_path = out.with_suffix(".json")
    if out.exists() and (args.split == "test" or not args.overwrite):
        p.error(f"{out} already exists" + (
            "; test outputs are never overwritten" if args.split == "test" else "; add --overwrite to replace it"))
    if out.resolve() == args.checkpoint.resolve():
        p.error("output and checkpoint paths must differ")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "multi_exit":
        p.error("Expected a multi-exit checkpoint produced by train.py; this is an older or unsupported model")
    data_config = checkpoint["data_config"]
    if data_config["dataset"] != "fashion-mnist":
        p.error("Unsupported checkpoint dataset")
    if (data_config.get("split_method"), data_config.get("preprocessing")) != (SPLIT_METHOD, PREPROCESSING):
        p.error("Checkpoint was trained with an older data split or pixel normalization; retrain it with train.py")
    try:
        device = select_device(args.device)
    except ValueError as error:
        p.error(str(error))

    data_dir = args.data_dir or Path(data_config["data_dir"])
    splits = get_image_splits(
        data_dir=data_dir, demo=data_config["demo"],
        val_size=data_config["val_size"], split_seed=data_config["split_seed"],
        download=not args.no_download,
    )
    if list(splits.class_names) != checkpoint["class_names"]:
        raise ValueError("Dataset class names differ from training")
    if args.split == "val":
        x, labels, image_indices = splits.val_x, splits.val_y, splits.val_indices
    else:
        x, labels, image_indices = splits.test_x, splits.test_y, np.arange(len(splits.test_y))

    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    logits = collect_logits(model, x, device, args.batch_size)
    metadata = {
        "split": args.split,
        "demo": data_config["demo"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "model_config": checkpoint["model_config"],
        "data_config": {**data_config, "data_dir": str(data_dir.resolve())},
        "training_config": checkpoint["training_config"],
        "class_names": list(splits.class_names),
        "evaluation_device": str(device),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_exit_outputs(out, logits, labels, image_indices, metadata)
    metrics = exit_metrics(logits, labels)
    summary = {**metadata, "outputs": str(out), "images": int(len(labels)), **metrics,
               "mean_macro_f1": float(np.mean(metrics["exit_macro_f1"]))}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(("SYNTHETIC IMAGE DEMO, " if data_config["demo"] else "") +
          f"{args.split} split: {len(labels)} images, checkpoint epoch {checkpoint['epoch']}, device {device}")
    print(" exit  accuracy  macro-F1    loss")
    for k, (acc, f1, loss) in enumerate(zip(metrics["exit_accuracy"], metrics["exit_macro_f1"], metrics["exit_loss"]), 1):
        print(f"  {k:2d}    {acc:.3f}     {f1:.3f}    {loss:.4f}")
    print("Saved:", out)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
