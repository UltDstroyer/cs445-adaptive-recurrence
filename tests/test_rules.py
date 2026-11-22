import json
from pathlib import Path

import numpy as np
import pytest
import torch

import evaluate
import logistic
import sweep
import train
from exit_outputs import ExitOutputs, load_exit_outputs
from model import MultiExitCNN
from rules import confidence_exits, first_stop, fixed_exit, never_stop, predictions_at, score

CNN_COUNTS = {"stem": 28_224, "blocks": [451_584] * 3 + [250_880] + [451_584] * 6, "heads": [160] * 3 + [320] * 7}


def test_first_stop_respects_min_exit_and_falls_back_to_the_last_exit():
    may_stop = np.array([
        [True, True, False, False],   # exit 1 is too early, so exit 2
        [False, False, True, True],   # exit 3
        [False, False, False, False],  # never marked: last exit
        [True, False, False, False],  # only exit 1 marked: last exit
    ])
    assert first_stop(may_stop).tolist() == [2, 3, 4, 4]
    assert first_stop(may_stop, min_exit=1).tolist() == [1, 3, 4, 1]
    with pytest.raises(ValueError):
        first_stop(may_stop, min_exit=5)


def test_confidence_rule_on_hand_written_probabilities():
    probabilities = np.array([
        [[0.95, 0.05], [0.90, 0.10], [0.60, 0.40]],  # confident at exit 1 (too early), 0.90 at exit 2
        [[0.60, 0.40], [0.70, 0.30], [0.55, 0.45]],  # never reaches 0.8
        [[0.50, 0.50], [0.10, 0.90], [0.05, 0.95]],  # reaches 0.9 at exit 2 (class 1)
    ])
    assert confidence_exits(probabilities, 0.8).tolist() == [2, 3, 2]
    assert confidence_exits(probabilities, 0.95).tolist() == [3, 3, 3]
    assert predictions_at(probabilities, np.array([2, 3, 2])).tolist() == [0, 0, 1]
    assert never_stop(3, 3).tolist() == [3, 3, 3]
    assert fixed_exit(3, 1).tolist() == [1, 1, 1]


def _random_outputs(seed=0, images=400):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(images, 10, 10)).astype(np.float32) * np.linspace(0.5, 4, 10)[None, :, None]
    labels = rng.integers(0, 10, images)
    return ExitOutputs(logits, labels, np.arange(images), {"split": "val", "multiply_adds": CNN_COUNTS})


def test_raising_the_threshold_never_lowers_compute():
    outputs = _random_outputs()
    probabilities = outputs.probabilities()
    computes = [score(outputs, confidence_exits(probabilities, q), probabilities)["compute"]
                for q in sweep.CONFIDENCE_THRESHOLDS]
    assert computes == sorted(computes)
    assert all(0.214 < c <= 1.0 for c in computes)  # nothing stops before exit 2 (21.4%)


def test_never_stopping_reproduces_the_last_exit_exactly():
    outputs = _random_outputs()
    result = sweep.run(outputs)
    assert result["never_stop"]["reproduces_last_exit"] is True
    assert result["never_stop"]["compute"] == 1.0
    assert result["never_stop"]["macro_f1"] == result["fixed_exits"][-1]["macro_f1"]
    # A threshold above every probability also never stops early.
    probabilities = outputs.probabilities()
    assert (confidence_exits(probabilities, 1.01) == 10).all()


def test_fixed_exits_pay_only_for_their_own_head():
    outputs = _random_outputs()
    result = sweep.run(outputs)
    fractions = outputs.compute_profile().fractions(all_heads=False)
    assert [row["compute"] for row in result["fixed_exits"]] == pytest.approx(fractions.tolist())


def test_thresholds_reach_very_close_to_one():
    assert sweep.CONFIDENCE_THRESHOLDS[0] == 0.1
    assert max(sweep.CONFIDENCE_THRESHOLDS) == pytest.approx(1 - 1e-7)
    assert list(sweep.CONFIDENCE_THRESHOLDS) == sorted(set(sweep.CONFIDENCE_THRESHOLDS))


def test_rules_on_saved_outputs_match_sequential_inference(tmp_path):
    """Applying the confidence rule to saved logits picks the same exits as running the network."""
    torch.manual_seed(0)
    model = MultiExitCNN().eval()
    x = torch.rand(64, 1, 28, 28)
    with torch.no_grad():
        saved = model(x).numpy()
    probabilities = ExitOutputs(saved, np.zeros(64, dtype=np.int64), np.arange(64), {}).probabilities()
    # The median exit-2 confidence: about half the images stop at exit 2, the rest later.
    threshold = float(np.quantile(probabilities.max(-1)[:, 1], 0.5))
    expected = confidence_exits(probabilities, threshold)
    _, exits = model.predict_sequential(
        x, lambda n, history: history[:, -1].softmax(-1).amax(-1) >= threshold, min_exit=2)
    assert exits.tolist() == expected.tolist()
    assert len(set(expected.tolist())) > 1  # the check covers several different exits


def test_sweep_and_logistic_on_the_demo(tmp_path):
    checkpoint = tmp_path / "demo.pt"
    train.main(["--demo", "--epochs", "2", "--device", "cpu", "--out", str(checkpoint)])
    outputs_path = tmp_path / "demo_val_outputs.npz"
    evaluate.main(["--checkpoint", str(checkpoint), "--device", "cpu", "--out", str(outputs_path)])
    logistic_path = tmp_path / "logistic_val_outputs.npz"
    logistic.main(["--demo", "--out", str(logistic_path)])
    sweep.main([str(outputs_path), "--logistic", str(logistic_path), "--inspect", "5"])

    result = json.loads((tmp_path / "demo_val_baselines.json").read_text())
    assert result["never_stop"]["reproduces_last_exit"] is True
    assert len(result["fixed_exits"]) == 10
    assert len(result["confidence_sweep"]) == len(sweep.CONFIDENCE_THRESHOLDS)
    assert len(result["inspection_sample"]) == 5
    lr = result["logistic_regression"]
    assert lr["multiply_adds"] == 7_840 and lr["compute"] == pytest.approx(7_840 / 4_346_080)
    assert lr["accuracy"] > 0.9  # the demo classes are easy to separate
    assert load_exit_outputs(logistic_path).logits.shape == (100, 1, 10)


def test_sweep_refuses_test_outputs(tmp_path):
    outputs = _random_outputs()
    path = tmp_path / "x_test_outputs.npz"
    from exit_outputs import save_exit_outputs
    save_exit_outputs(path, outputs.logits, outputs.labels, outputs.image_indices,
                      {"split": "test", "multiply_adds": CNN_COUNTS})
    with pytest.raises(SystemExit):
        sweep.main([str(path)])
    sweep.main([str(path), "--final-test", "--out", str(tmp_path / "final.json"), "--inspect", "1"])
    assert Path(tmp_path / "final.json").exists()


def test_logistic_regression_starts_from_zeros_and_is_deterministic():
    rng = np.random.default_rng(0)
    x = rng.random((200, 1, 4, 4)).astype(np.float32)
    y = (x.reshape(200, -1)[:, 0] > 0.5).astype(np.int64)
    first = logistic.fit_logistic_regression(x, y, num_classes=2, max_iterations=50)
    second = logistic.fit_logistic_regression(x, y, num_classes=2, max_iterations=50)
    np.testing.assert_array_equal(first[0], second[0])
    predictions = (x.reshape(200, -1) @ first[0] + first[1]).argmax(1)
    assert (predictions == y).mean() > 0.95
