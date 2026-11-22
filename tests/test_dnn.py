import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import data
import evaluate
import train
from model import DNNClassifier


def test_dnn_dense_layers_have_independent_weights_and_learn():
    model = DNNClassifier(hidden_sizes=(16, 16), dropout=0)
    dense = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in dense] == [(784, 16), (16, 16), (16, 10)]
    assert len({layer.weight.data_ptr() for layer in dense}) == 3
    recurrent_types = (nn.RNN, nn.GRU, nn.LSTM, nn.RNNCell, nn.GRUCell, nn.LSTMCell)
    assert not any(isinstance(layer, recurrent_types) for layer in model.modules())
    logits = model(torch.randn(12, 1, 28, 28))
    assert logits.shape == (12, 10)
    F.cross_entropy(logits, torch.arange(12) % 10).backward()
    assert all(layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all() for layer in dense)


def test_image_split_is_stratified_disjoint_reproducible_and_scaled_only():
    images = np.broadcast_to(np.arange(60, dtype=np.uint8)[:, None, None], (60, 28, 28)).copy()
    labels = np.arange(60, dtype=np.int64) % 10
    test_images = np.full((20, 28, 28), 255, dtype=np.uint8)
    test_labels = np.arange(20, dtype=np.int64) % 10
    splits = data.split_images(images, labels, test_images, test_labels, val_size=20, split_seed=17)
    assert splits.train_x.shape == (40, 1, 28, 28)
    assert splits.val_x.shape == (20, 1, 28, 28)
    assert splits.test_x.shape == (20, 1, 28, 28)
    assert np.bincount(splits.val_y, minlength=10).tolist() == [2] * 10
    assert np.bincount(splits.train_y, minlength=10).tolist() == [4] * 10
    assert not np.intersect1d(splits.train_indices, splits.val_indices).size
    np.testing.assert_array_equal(
        np.sort(np.concatenate([splits.train_indices, splits.val_indices])), np.arange(60)
    )
    np.testing.assert_array_equal(splits.val_y, labels[splits.val_indices])
    np.testing.assert_array_equal(splits.train_x, images[splits.train_indices, None] / np.float32(255))
    np.testing.assert_array_equal(splits.test_x, 1.0)
    again = data.split_images(images, labels, test_images, test_labels, val_size=20, split_seed=17)
    np.testing.assert_array_equal(again.val_indices, splits.val_indices)
    other = data.split_images(images, labels, test_images, test_labels, val_size=20, split_seed=18)
    assert not np.array_equal(other.val_indices, splits.val_indices)


def test_stratified_split_keeps_class_proportions_for_uneven_classes():
    labels = np.repeat(np.arange(3), [50, 30, 20])
    train_indices, val_indices = data.stratified_split_indices(labels, val_size=10, split_seed=0)
    assert np.bincount(labels[val_indices]).tolist() == [5, 3, 2]
    assert len(train_indices) == 90
    _, rounded = data.stratified_split_indices(labels, val_size=7, split_seed=0)
    assert len(rounded) == 7


def test_default_split_reserves_600_validation_images_per_class(monkeypatch):
    images = np.zeros((60000, 28, 28), dtype=np.uint8)
    labels = np.arange(60000, dtype=np.int64) % 10
    monkeypatch.setattr(
        data, "load_fashion_mnist", lambda *args: (images, labels, images[:100], labels[:100])
    )
    splits = data.get_image_splits()
    assert np.bincount(splits.val_y, minlength=10).tolist() == [600] * 10
    assert np.bincount(splits.train_y, minlength=10).tolist() == [5400] * 10


def test_split_summary_is_saved_without_test_pixel_statistics(tmp_path):
    out = tmp_path / "summary.json"
    data.main(["--demo", "--summary-out", str(out)])
    summary = json.loads(out.read_text())
    assert summary["split_method"] == data.SPLIT_METHOD
    assert summary["preprocessing"] == data.PREPROCESSING
    assert summary["train_val_disjoint"] is True
    assert summary["splits"]["val"]["class_counts"] == [10] * 10
    assert summary["splits"]["train"]["images"] == 500
    assert 0 <= summary["splits"]["train"]["pixel_min"] <= summary["splits"]["train"]["pixel_max"] <= 1
    assert "pixel_mean" not in summary["splits"]["test"]


def test_real_data_path_passes_cache_and_download_settings(monkeypatch, tmp_path):
    calls = []

    def load(directory, download):
        calls.append((directory, download))
        return data.demo_images()

    monkeypatch.setattr(data, "load_fashion_mnist", load)
    splits = data.get_image_splits(tmp_path, val_size=100, download=False)
    assert calls == [(tmp_path, False)]
    assert len(splits.train_y) == 500 and len(splits.test_y) == 100


@pytest.mark.parametrize("val_size", [0, 599, 600, 601])
def test_invalid_image_split_sizes(val_size):
    with pytest.raises(ValueError, match="val_size"):
        data.split_images(*data.demo_images(), val_size=val_size)


def test_training_records_split_settings_and_evaluation_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_fashion_mnist", lambda *args: pytest.fail("demo attempted download"))
    checkpoint_path = tmp_path / "multi_exit.pt"
    output_path = tmp_path / "evaluation.json"
    train.main([
        "--demo", "--epochs", "1", "--split-seed", "17", "--val-size", "120",
        "--device", "cpu", "--out", str(checkpoint_path),
    ])
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["data_config"]["split_seed"] == 17
    assert checkpoint["data_config"]["val_size"] == 120
    assert checkpoint["data_config"]["split_method"] == data.SPLIT_METHOD
    assert checkpoint["data_config"]["preprocessing"] == data.PREPROCESSING
    assert "pixel_mean" not in checkpoint
    # Evaluation of multi-exit checkpoints arrives with separate validation/test runs.
    monkeypatch.setattr(evaluate, "get_image_splits", lambda **kwargs: pytest.fail("evaluation loaded images"))
    with pytest.raises(SystemExit):
        evaluate.main(["--checkpoint", str(checkpoint_path), "--out", str(output_path)])
    assert not output_path.exists()


def test_evaluation_confusion_and_macro_f1():
    class Predictions(nn.Module):
        num_classes = 3

        def forward(self, x):
            return x

    logits = np.array([[10, 0, 0], [0, 10, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float32)
    metrics = evaluate.evaluate_images(Predictions(), logits, np.array([0, 0, 1, 2]), torch.device("cpu"), 3)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_legacy_checkpoint_is_rejected_before_loading_images(tmp_path, monkeypatch):
    path = tmp_path / "old.pt"
    torch.save({"model_state": {}}, path)
    monkeypatch.setattr(evaluate, "get_image_splits", lambda **kwargs: pytest.fail("legacy checkpoint loaded images"))
    with pytest.raises(SystemExit):
        evaluate.main(["--checkpoint", str(path)])


def test_checkpoint_with_old_split_or_normalization_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "standardized.pt"
    torch.save({
        "model_type": "dnn",
        "data_config": {"dataset": "fashion-mnist", "demo": True, "data_dir": str(tmp_path),
                        "val_size": 100, "split_seed": 7},
        "pixel_mean": 0.3, "pixel_std": 0.35,
    }, path)
    monkeypatch.setattr(evaluate, "get_image_splits", lambda **kwargs: pytest.fail("old checkpoint loaded images"))
    with pytest.raises(SystemExit):
        evaluate.main(["--checkpoint", str(path), "--out", str(tmp_path / "out.json")])
