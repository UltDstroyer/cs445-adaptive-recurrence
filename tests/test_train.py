import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import train
from model import MultiExitClassifier, MultiExitCNN, build_model


def test_mlp_can_still_be_trained(tmp_path):
    checkpoint_path = tmp_path / "mlp.pt"
    train.main(["--demo", "--epochs", "1", "--architecture", "mlp", "--device", "cpu", "--out", str(checkpoint_path)])
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["model_config"] == {"architecture": "mlp", "input_shape": [1, 28, 28], "width": 32,
                                          "block_hidden": 128, "num_exits": 10, "num_classes": 10}
    model = build_model(checkpoint["model_config"])
    assert isinstance(model, MultiExitClassifier)
    model.load_state_dict(checkpoint["model_state"])


def test_training_defaults_and_history_round_trip(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    history_path = tmp_path / "training.json"
    train.main([
        "--demo", "--epochs", "2", "--device", "cpu",
        "--out", str(checkpoint_path), "--history", str(history_path),
    ])
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    config = checkpoint["model_config"]
    assert checkpoint["model_type"] == "multi_exit"
    assert config == {"architecture": "cnn", "input_shape": [1, 28, 28], "channels": [16, 32],
                      "downsample_at": 4, "num_exits": 10, "num_classes": 10}
    assert checkpoint["training_config"]["lr"] == 0.001
    assert checkpoint["training_config"]["seed"] == 0
    assert checkpoint["data_config"]["dataset"] == "fashion-mnist"
    model = build_model(config)
    assert isinstance(model, MultiExitCNN)
    model.load_state_dict(checkpoint["model_state"])
    assert len(history["history"]) == history["epochs_completed"] == 2
    assert history["training_seconds"] > 0
    chosen = max(history["history"], key=lambda row: (row["val_mean_macro_f1"], -row["val_loss"]))
    assert checkpoint["epoch"] == history["best_epoch"] == chosen["epoch"]
    assert checkpoint["val_mean_macro_f1"] == chosen["val_mean_macro_f1"]
    for row in history["history"]:
        assert len(row["val_exit_macro_f1"]) == len(row["val_exit_accuracy"]) == 10
        assert row["val_mean_macro_f1"] == pytest.approx(np.mean(row["val_exit_macro_f1"]))
        assert all(0 <= v <= 1 for v in row["val_exit_macro_f1"] + row["val_exit_accuracy"])
        assert row["train_loss"] >= 0 and row["val_loss"] >= 0 and row["seconds"] > 0


def test_selection_uses_mean_exit_macro_f1_then_loss_and_patience(tmp_path, monkeypatch):
    def metrics(mean_f1, loss):
        return {"loss": loss, "mean_macro_f1": mean_f1, "exit_loss": [loss] * 10,
                "exit_accuracy": [mean_f1] * 10, "exit_macro_f1": [mean_f1] * 10}

    results = iter([metrics(0.6, 0.8), metrics(0.6, 0.7), metrics(0.55, 0.5)])
    monkeypatch.setattr(train, "evaluate_split", lambda *args, **kwargs: next(results))
    checkpoint_path = tmp_path / "model.pt"
    history_path = tmp_path / "training.json"
    train.main([
        "--demo", "--epochs", "10", "--patience", "1", "--batch-size", "144",
        "--device", "cpu", "--out", str(checkpoint_path), "--history", str(history_path),
    ])
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert checkpoint["epoch"] == history["best_epoch"] == 2
    assert checkpoint["val_loss"] == 0.7
    assert history["epochs_completed"] == 3 and history["stopped_early"]


def test_loss_is_the_average_of_per_exit_cross_entropies():
    torch.manual_seed(0)
    logits = torch.randn(5, 4, 3)
    labels = torch.tensor([0, 1, 2, 1, 0])
    per_exit = [F.cross_entropy(logits[:, k], labels) for k in range(4)]
    torch.testing.assert_close(train.multi_exit_loss(logits, labels), torch.stack(per_exit).mean())


def test_validation_metrics_are_reported_for_each_exit():
    class FixedExits(torch.nn.Module):
        num_exits, num_classes = 2, 3

        def forward(self, x):
            # Exit 1 predicts the class stored in the image; exit 2 always predicts class 0.
            exit_1 = F.one_hot(x[:, 0].long(), 3).float() * 10
            exit_2 = F.one_hot(torch.zeros(len(x), dtype=torch.long), 3).float() * 10
            return torch.stack([exit_1, exit_2], dim=1)

    labels = np.array([0, 1, 2, 2], dtype=np.int64)
    images = labels[:, None].astype(np.float32)
    result = train.evaluate_split(FixedExits(), images, labels, torch.device("cpu"), batch_size=3)
    assert result["exit_accuracy"] == [1.0, 0.25]
    assert result["exit_macro_f1"][0] == 1.0
    assert result["exit_macro_f1"][1] == pytest.approx((0.4 + 0 + 0) / 3)
    assert result["mean_macro_f1"] == pytest.approx(np.mean(result["exit_macro_f1"]))
    assert result["exit_loss"][0] < result["exit_loss"][1]


def _without_timing(rows):
    return [{k: v for k, v in row.items() if k != "seconds"} for row in rows]


def test_interrupted_run_resumes_to_the_same_result(tmp_path, monkeypatch):
    common = ["--demo", "--epochs", "3", "--device", "cpu", "--threads", "1"]
    straight = tmp_path / "straight.pt"
    train.main(common + ["--out", str(straight)])

    resumed = tmp_path / "resumed.pt"
    real_evaluate, calls = train.evaluate_split, []

    def interrupt_in_epoch_2(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(train, "evaluate_split", interrupt_in_epoch_2)
    with pytest.raises(SystemExit) as stopped:
        train.main(common + ["--out", str(resumed)])
    assert stopped.value.code == 130
    state = torch.load(train.resume_path(resumed), weights_only=True)
    assert state["epoch"] == 1
    monkeypatch.setattr(train, "evaluate_split", real_evaluate)
    train.main(common + ["--out", str(resumed), "--resume"])

    a = json.loads(straight.with_name("straight_training.json").read_text())
    b = json.loads(resumed.with_name("resumed_training.json").read_text())
    assert _without_timing(a["history"]) == _without_timing(b["history"])
    assert a["best_epoch"] == b["best_epoch"]
    first, second = (torch.load(p, weights_only=True)["model_state"] for p in (straight, resumed))
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_resume_extends_a_finished_run_and_rejects_changed_settings(tmp_path):
    out = tmp_path / "run.pt"
    common = ["--demo", "--device", "cpu", "--out", str(out)]
    train.main(common + ["--epochs", "1"])
    train.main(common + ["--epochs", "2", "--resume"])
    history = json.loads(out.with_name("run_training.json").read_text())
    assert [row["epoch"] for row in history["history"]] == [1, 2]
    with pytest.raises(SystemExit):
        train.main(common + ["--epochs", "3", "--resume", "--lr", "0.01"])
    with pytest.raises(SystemExit):
        train.main(["--demo", "--resume", "--out", str(tmp_path / "never_started.pt")])


def test_device_selection_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert train.select_device("auto").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train, "mps_available", lambda: True)
    assert train.select_device("auto").type == "mps"
    monkeypatch.setattr(train, "mps_available", lambda: False)
    assert train.select_device("auto").type == "cpu"
    with pytest.raises(ValueError, match="no Apple-silicon GPU"):
        train.select_device("mps")
    with pytest.raises(ValueError, match="no NVIDIA GPU"):
        train.select_device("cuda")


def test_default_training_length_is_30_epochs():
    assert train.DEFAULT_EPOCHS == 30


@pytest.mark.parametrize("arguments", [
    ["--width", "0"], ["--channels", "16", "0"], ["--downsample-at", "1"],
    ["--patience", "-1"], ["--lr", "nan"], ["--threads", "0"],
])
def test_invalid_training_options_are_rejected_before_loading_data(arguments, monkeypatch):
    monkeypatch.setattr(train, "get_image_splits", lambda *args, **kwargs: pytest.fail("invalid options loaded data"))
    with pytest.raises(SystemExit):
        train.main(arguments)
