import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import train
from model import MultiExitClassifier


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
    assert config == {"input_shape": [1, 28, 28], "width": 32, "block_hidden": 128,
                      "num_exits": 10, "num_classes": 10}
    assert checkpoint["training_config"]["lr"] == 0.001
    assert checkpoint["training_config"]["seed"] == 0
    assert checkpoint["data_config"]["dataset"] == "fashion-mnist"
    model = MultiExitClassifier(**config)
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


@pytest.mark.parametrize("arguments", [["--width", "0"], ["--patience", "-1"], ["--lr", "nan"]])
def test_invalid_training_options_are_rejected_before_loading_data(arguments, monkeypatch):
    monkeypatch.setattr(train, "get_image_splits", lambda *args, **kwargs: pytest.fail("invalid options loaded data"))
    with pytest.raises(SystemExit):
        train.main(arguments)
