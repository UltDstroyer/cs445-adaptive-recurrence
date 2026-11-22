import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import data
import evaluate
import train
from exit_outputs import exit_metrics, load_exit_outputs, macro_f1, save_exit_outputs

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A small demo checkpoint trained on the CPU."""
    checkpoint = tmp_path_factory.mktemp("run") / "demo.pt"
    train.main(["--demo", "--epochs", "2", "--device", "cpu", "--split-seed", "17",
                "--val-size", "120", "--out", str(checkpoint)])
    return checkpoint


def test_validation_outputs_are_saved_for_every_exit(trained, tmp_path):
    out = tmp_path / "val.npz"
    evaluate.main(["--checkpoint", str(trained), "--device", "cpu", "--out", str(out)])
    outputs = load_exit_outputs(out)
    splits = data.get_image_splits(demo=True, val_size=120, split_seed=17)
    assert outputs.logits.shape == (120, 10, 10) and outputs.logits.dtype == np.float32
    np.testing.assert_array_equal(outputs.labels, splits.val_y)
    np.testing.assert_array_equal(outputs.image_indices, splits.val_indices)
    assert outputs.metadata["split"] == "val"
    assert outputs.metadata["model_config"]["architecture"] == "cnn"
    assert outputs.metadata["evaluation_device"] == "cpu"
    assert outputs.compute_profile().full == 4_346_080  # the demo trains the default CNN

    # The saved outputs reproduce the validation scores recorded during training.
    checkpoint = torch.load(trained, weights_only=True)
    metrics = exit_metrics(outputs.logits, outputs.labels)
    assert metrics["exit_macro_f1"] == pytest.approx(checkpoint["val_exit_macro_f1"], abs=1e-12)
    assert metrics["exit_accuracy"] == pytest.approx(checkpoint["val_exit_accuracy"], abs=1e-12)
    summary = json.loads(out.with_suffix(".json").read_text())
    assert summary["images"] == 120 and summary["exit_macro_f1"] == metrics["exit_macro_f1"]


def test_saved_outputs_load_without_pytorch(trained, tmp_path):
    out = tmp_path / "val.npz"
    evaluate.main(["--checkpoint", str(trained), "--device", "cpu", "--out", str(out)])
    script = (
        "import sys; sys.modules['torch'] = None\n"  # any torch import would now fail
        "from exit_outputs import load_exit_outputs, exit_metrics\n"
        f"o = load_exit_outputs({str(out)!r})\n"
        "m = exit_metrics(o.logits, o.labels)\n"
        "assert o.probabilities().shape == (120, 10, 10)\n"
        "assert round(o.compute_profile().fractions()[1], 3) == 0.214\n"
        "print(len(m['exit_macro_f1']))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "10"


def test_test_split_needs_confirmation_and_is_never_overwritten(trained, tmp_path, monkeypatch):
    out = tmp_path / "test.npz"
    real_splits = evaluate.get_image_splits
    monkeypatch.setattr(evaluate, "get_image_splits", lambda **kwargs: pytest.fail("loaded images"))
    with pytest.raises(SystemExit):
        evaluate.main(["--checkpoint", str(trained), "--split", "test", "--out", str(out)])
    monkeypatch.setattr(evaluate, "get_image_splits", real_splits)

    evaluate.main(["--checkpoint", str(trained), "--split", "test", "--confirm-final-test",
                   "--device", "cpu", "--out", str(out)])
    outputs = load_exit_outputs(out)
    assert outputs.metadata["split"] == "test" and outputs.logits.shape == (100, 10, 10)
    np.testing.assert_array_equal(outputs.image_indices, np.arange(100))

    monkeypatch.setattr(evaluate, "get_image_splits", lambda **kwargs: pytest.fail("loaded images"))
    for extra in ([], ["--overwrite"]):
        with pytest.raises(SystemExit):
            evaluate.main(["--checkpoint", str(trained), "--split", "test", "--confirm-final-test",
                           "--out", str(out), *extra])


def test_validation_output_needs_overwrite_to_be_replaced(trained, tmp_path):
    out = tmp_path / "val.npz"
    evaluate.main(["--checkpoint", str(trained), "--device", "cpu", "--out", str(out)])
    with pytest.raises(SystemExit):
        evaluate.main(["--checkpoint", str(trained), "--device", "cpu", "--out", str(out)])
    evaluate.main(["--checkpoint", str(trained), "--device", "cpu", "--out", str(out), "--overwrite"])


def test_default_output_path_names_the_checkpoint_and_split():
    assert evaluate.default_output(Path("checkpoints/cnn_seed3.pt"), "val") == Path("results/cnn_seed3_val_outputs.npz")


@pytest.mark.parametrize("saved", [
    {"model_state": {}},  # pre-multi-exit checkpoints
    {"model_type": "dnn", "data_config": {}},
    {"model_type": "multi_exit", "data_config": {"dataset": "fashion-mnist", "demo": True,
                                                 "data_dir": ".", "val_size": 100, "split_seed": 7}},
])
def test_older_checkpoints_are_rejected_before_loading_images(saved, tmp_path, monkeypatch):
    path = tmp_path / "old.pt"
    torch.save(saved, path)
    monkeypatch.setattr(evaluate, "get_image_splits", lambda **kwargs: pytest.fail("loaded images"))
    with pytest.raises(SystemExit):
        evaluate.main(["--checkpoint", str(path), "--out", str(tmp_path / "out.npz")])


def test_exit_metrics_and_macro_f1_on_known_values():
    labels = np.array([0, 0, 1, 2])
    # Exit 1 is right on 3 of 4 images (one class-0 image called class 1); exit 2 is always right.
    exit_1 = np.array([[10, 0, 0], [0, 10, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float32)
    exit_2 = np.eye(3, dtype=np.float32)[labels] * 10
    metrics = exit_metrics(np.stack([exit_1, exit_2], axis=1), labels)
    assert metrics["exit_accuracy"] == [0.75, 1.0]
    assert metrics["exit_macro_f1"][0] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)
    assert metrics["exit_macro_f1"][1] == 1.0
    assert metrics["exit_loss"][1] < metrics["exit_loss"][0]
    assert macro_f1(labels, np.array([0, 0, 1, 2]), 4) == pytest.approx(3 / 4)  # unused class 3 counts as 0


def test_invalid_outputs_are_rejected(tmp_path):
    good = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="labels"):
        save_exit_outputs(tmp_path / "a.npz", good, [0, 9], [0, 1], {})
    with pytest.raises(ValueError, match="repeat"):
        save_exit_outputs(tmp_path / "b.npz", good, [0, 1], [5, 5], {})
    with pytest.raises(ValueError, match="finite"):
        save_exit_outputs(tmp_path / "c.npz", good + np.nan, [0, 1], [0, 1], {})
