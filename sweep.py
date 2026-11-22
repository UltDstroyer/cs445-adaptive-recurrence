"""Baselines and the confidence-only sweep, from saved validation outputs (numpy only).

Reports, for one multi-exit network's saved outputs:
  * every fixed exit used on its own,
  * never stopping early (must reproduce the last exit exactly),
  * confidence-only stopping (Kaya et al.) for thresholds from 0.10 up to
    1 - 1e-7, from exit 2 onward,
  * optionally the logistic-regression baseline from logistic.py,
and prints a random sample of validation images with every exit's prediction
so the actual predictions can be inspected.

Writes results/<outputs name>_baselines.json. Outputs from the test split are
refused unless --final-test is given (step 13 only).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from exit_outputs import load_exit_outputs
from rules import MIN_EXIT, confidence_exits, fixed_exit, never_stop, predictions_at, score

# Evenly spaced thresholds, then thresholds ever closer to 1: exits are often
# overconfident, so most of the useful range can sit above 0.99.
CONFIDENCE_THRESHOLDS = tuple(
    [round(q, 2) for q in np.arange(0.10, 0.951, 0.05)]
    + [round(1 - 10 ** -k, 8) for k in np.arange(1.5, 7.01, 0.5)]
)


def run(outputs, logistic=None) -> dict:
    probabilities = outputs.probabilities()
    n, num_exits = len(outputs.labels), outputs.num_exits
    fixed = [{"exit": k, **score(outputs, fixed_exit(n, k), probabilities, all_heads=False)}
             for k in range(1, num_exits + 1)]
    never = never_stop(n, num_exits)
    never_predictions = predictions_at(probabilities, never)
    reproduces_last_exit = bool(
        (never == num_exits).all()
        and np.array_equal(never_predictions, outputs.logits[:, -1].argmax(axis=-1))
    )
    sweep = [{"threshold": q, **score(outputs, confidence_exits(probabilities, q), probabilities)}
             for q in CONFIDENCE_THRESHOLDS]
    result = {
        "outputs_metadata": {k: outputs.metadata.get(k) for k in
                             ("split", "demo", "checkpoint", "checkpoint_epoch", "model_config", "evaluation_device")},
        "images": n,
        "min_exit": MIN_EXIT,
        "fixed_exits": fixed,
        "never_stop": {**score(outputs, never, probabilities), "reproduces_last_exit": reproduces_last_exit},
        "confidence_sweep": sweep,
    }
    if logistic is not None:
        if not np.array_equal(logistic.image_indices, outputs.image_indices):
            raise ValueError("The logistic-regression outputs cover different validation images")
        logistic_exits = fixed_exit(n, 1)
        logistic_score = score(logistic, logistic_exits)
        logistic_score["compute"] = logistic.compute_profile().full / outputs.compute_profile().full
        logistic_score["multiply_adds"] = logistic.compute_profile().full
        result["logistic_regression"] = logistic_score
    return result


def inspection_sample(outputs, count: int, threshold: float, seed: int) -> list[dict]:
    """A random sample of images with every exit's prediction and confidence."""
    probabilities = outputs.probabilities()
    rows = np.random.default_rng(seed).choice(len(outputs.labels), size=min(count, len(outputs.labels)), replace=False)
    stops = confidence_exits(probabilities, threshold)
    names = outputs.metadata.get("class_names") or [str(c) for c in range(outputs.num_classes)]
    sample = []
    for row in sorted(rows):
        predicted = probabilities[row].argmax(axis=-1)
        sample.append({
            "image_index": int(outputs.image_indices[row]),
            "label": names[outputs.labels[row]],
            "exit_predictions": [names[c] for c in predicted],
            "exit_confidence": [round(float(c), 3) for c in probabilities[row].max(axis=-1)],
            "confidence_rule_exit": int(stops[row]),
            "confidence_rule_correct": bool(predicted[stops[row] - 1] == outputs.labels[row]),
        })
    return sample


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outputs", type=Path, help="Validation outputs from evaluate.py (.npz)")
    p.add_argument("--logistic", type=Path, help="Validation outputs from logistic.py (.npz)")
    p.add_argument("--out", type=Path, help="Results JSON (default: next to the outputs, *_baselines.json)")
    p.add_argument("--inspect", type=int, default=12, help="How many random validation images to print")
    p.add_argument("--inspect-threshold", type=float, default=0.9,
                   help="Confidence threshold used to show where each sampled image would stop")
    p.add_argument("--seed", type=int, default=0, help="Which random images to print")
    p.add_argument("--final-test", action="store_true", help="Allow test-split outputs (final evaluation only)")
    args = p.parse_args(argv)
    outputs = load_exit_outputs(args.outputs)
    if outputs.metadata.get("split") != "val" and not args.final_test:
        p.error("these outputs are not from the validation split; tuning and baselines use validation only")
    logistic = load_exit_outputs(args.logistic) if args.logistic else None
    result = run(outputs, logistic)
    result["inspection_sample"] = inspection_sample(outputs, args.inspect, args.inspect_threshold, args.seed)
    result["inspection_threshold"] = args.inspect_threshold
    out = args.out or args.outputs.with_name(args.outputs.stem.removesuffix("_outputs") + "_baselines.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    split = outputs.metadata.get("split")
    print(("SYNTHETIC IMAGE DEMO, " if outputs.metadata.get("demo") else "") + f"{split} split, {len(outputs.labels)} images")
    print("\nFixed exits (each used on its own)")
    print(" exit  compute  macro-F1  accuracy")
    for row in result["fixed_exits"]:
        print(f"  {row['exit']:2d}    {row['compute']:.3f}    {row['macro_f1']:.3f}     {row['accuracy']:.3f}")
    never = result["never_stop"]
    print(f"\nNever stop early: compute {never['compute']:.3f}, macro-F1 {never['macro_f1']:.3f}; "
          f"reproduces the last exit: {never['reproduces_last_exit']}")
    if "logistic_regression" in result:
        lr = result["logistic_regression"]
        print(f"Logistic regression on pixels: macro-F1 {lr['macro_f1']:.3f}, accuracy {lr['accuracy']:.3f}, "
              f"compute {lr['compute']:.4f} of the network ({lr['multiply_adds']:,} multiply-adds)")
    print(f"\nConfidence only, from exit {MIN_EXIT}")
    print(" threshold     compute  macro-F1  accuracy  images per exit")
    for row in result["confidence_sweep"]:
        print(f" {row['threshold']:<12}  {row['compute']:.3f}    {row['macro_f1']:.3f}     "
              f"{row['accuracy']:.3f}     {row['exit_counts']}")
    print(f"\nSample of {len(result['inspection_sample'])} validation images "
          f"(stop = confidence rule at {args.inspect_threshold}):")
    for item in result["inspection_sample"]:
        stop = item["confidence_rule_exit"]
        marks = " ".join(f"{p[:7]:<7}" for p in item["exit_predictions"])
        print(f" #{item['image_index']:<6} true {item['label']:<12} stop {stop:2d} "
              f"{'right' if item['confidence_rule_correct'] else 'WRONG'} | exits 1-10: {marks}")
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
