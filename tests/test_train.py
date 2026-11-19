import json

import pytest
import torch

import train
from model import DEFAULT_DROPOUT, DEFAULT_HIDDEN_SIZES


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
    assert checkpoint["model_type"] == "dnn"
    assert config["input_shape"] == [1, 28, 28]
    assert config["hidden_sizes"] == list(DEFAULT_HIDDEN_SIZES) == [256, 128]
    assert config["num_classes"] == 10
    assert config["dropout"] == DEFAULT_DROPOUT == 0.2
    assert checkpoint["training_config"]["lr"] == 0.001
    assert checkpoint["data_config"]["dataset"] == "fashion-mnist"
    assert len(history["history"]) == history["epochs_completed"] == 2
    chosen = max(history["history"], key=lambda row: (row["val_accuracy"], -row["val_loss"]))
    assert checkpoint["epoch"] == history["best_epoch"] == chosen["epoch"]
    assert checkpoint["val_accuracy"] == chosen["val_accuracy"]
    for row in history["history"]:
        assert 0 <= row["train_accuracy"] <= 1 and 0 <= row["val_accuracy"] <= 1
        assert row["train_loss"] >= 0 and row["val_loss"] >= 0


def test_stopping_and_accuracy_ties_use_only_validation_metrics(tmp_path, monkeypatch):
    metrics = iter([
        {"loss": 0.8, "accuracy": 0.6},
        {"loss": 0.7, "accuracy": 0.6},
        {"loss": 0.5, "accuracy": 0.55},
    ])
    monkeypatch.setattr(train, "evaluate_split", lambda *args, **kwargs: next(metrics))
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


@pytest.mark.parametrize("arguments", [["--dropout", "1"], ["--patience", "-1"], ["--lr", "nan"]])
def test_invalid_training_options_are_rejected_before_loading_data(arguments, monkeypatch):
    monkeypatch.setattr(train, "get_image_splits", lambda *args, **kwargs: pytest.fail("invalid options loaded data"))
    with pytest.raises(SystemExit):
        train.main(arguments)
