"""Early-exit stopping rules applied to saved per-exit outputs (numpy only).

A rule looks at the class probabilities from every exit and marks the exits
where each image may stop. Each image stops at the first marked exit from
MIN_EXIT onward; an image that is never marked stops at the last exit. Exits are
numbered from 1. Because rules only use exits up to where they stop, applying
them to saved outputs gives the same answer as running the network sequentially.
"""

import numpy as np

from exit_outputs import ExitOutputs, macro_f1

MIN_EXIT = 2  # every adaptive rule may stop from exit 2 onward (plan, section 1.2)


def first_stop(may_stop: np.ndarray, min_exit: int = MIN_EXIT) -> np.ndarray:
    """Exit number (1-based) where each image stops, from a [images, exits] boolean array."""
    may_stop = np.array(may_stop, dtype=bool)
    if may_stop.ndim != 2 or not 1 <= min_exit <= may_stop.shape[1]:
        raise ValueError("may_stop must be [images, exits] and min_exit a valid exit number")
    may_stop[:, :min_exit - 1] = False
    may_stop[:, -1] = True
    return may_stop.argmax(axis=1) + 1


def fixed_exit(num_images: int, exit_number: int) -> np.ndarray:
    """Every image uses the same exit."""
    return np.full(num_images, exit_number, dtype=np.int64)


def confidence_exits(probabilities: np.ndarray, threshold: float, min_exit: int = MIN_EXIT) -> np.ndarray:
    """Confidence only (Kaya et al.): stop once the top class probability reaches the threshold."""
    return first_stop(probabilities.max(axis=-1) >= threshold, min_exit)


def never_stop(num_images: int, num_exits: int) -> np.ndarray:
    """Never stop early: every image runs to the last exit."""
    return first_stop(np.zeros((num_images, num_exits), dtype=bool))


def predictions_at(probabilities: np.ndarray, exits: np.ndarray) -> np.ndarray:
    """The class each image's chosen exit predicts."""
    return probabilities[np.arange(len(exits)), exits - 1].argmax(axis=-1)


def score(outputs: ExitOutputs, exits: np.ndarray, probabilities: np.ndarray | None = None,
          all_heads: bool = True) -> dict:
    """Macro-F1, accuracy and average compute (share of the full model) for chosen exits.

    all_heads=True charges for every head up to the stop, as an adaptive rule
    needs; use all_heads=False for a fixed exit used on its own.
    """
    probabilities = outputs.probabilities() if probabilities is None else probabilities
    predicted = predictions_at(probabilities, exits)
    return {
        "macro_f1": macro_f1(outputs.labels, predicted, outputs.num_classes),
        "accuracy": float((predicted == outputs.labels).mean()),
        "compute": outputs.compute_profile().mean_fraction(exits, all_heads=all_heads),
        "exit_counts": np.bincount(exits, minlength=outputs.num_exits + 1)[1:].tolist(),
    }
