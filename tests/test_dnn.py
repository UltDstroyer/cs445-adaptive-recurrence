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


def test_image_split_is_reproducible_disjoint_and_uses_train_statistics_only():
    images = np.broadcast_to(np.arange(60, dtype=np.uint8)[:, None, None], (60, 28, 28)).copy()
    labels = np.arange(60, dtype=np.int64) % 10
    test_images = np.full((20, 28, 28), 255, dtype=np.uint8)
    test_labels = np.arange(20, dtype=np.int64) % 10
    splits = data.split_images(images, labels, test_images, test_labels, val_size=10, split_seed=17)
    indices = np.random.default_rng(17).permutation(60)
    train_indices, val_indices = indices[10:], indices[:10]
    expected_train = images[train_indices, None].astype(np.float32) / 255
    assert splits.train_x.shape == (50, 1, 28, 28)
    assert splits.val_x.shape == (10, 1, 28, 28)
    assert splits.test_x.shape == (20, 1, 28, 28)
    assert splits.mean == pytest.approx(expected_train.mean(dtype=np.float64))
    assert splits.std == pytest.approx(expected_train.std(dtype=np.float64))
    np.testing.assert_allclose(splits.train_x.mean(), 0, atol=1e-6)
    np.testing.assert_allclose(splits.train_x.std(), 1, atol=1e-6)
    np.testing.assert_array_equal(splits.train_y, labels[train_indices])
    np.testing.assert_array_equal(splits.val_y, labels[val_indices])
    restored = data.split_images(
        images, labels, test_images, test_labels, val_size=10, split_seed=17,
        stats=(splits.mean, splits.std),
    )
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(getattr(splits, f"{name}_x"), getattr(restored, f"{name}_x"))
    supplied = data.split_images(images, labels, test_images, test_labels, val_size=10, stats=(0.25, 0.5))
    np.testing.assert_allclose(supplied.test_x, 1.5)


def test_real_data_path_passes_cache_and_download_settings(monkeypatch, tmp_path):
    calls = []

    def load(directory, download):
        calls.append((directory, download))
        return data.demo_images()

    monkeypatch.setattr(data, "load_fashion_mnist", load)
    splits = data.get_image_splits(tmp_path, val_size=100, download=False)
    assert calls == [(tmp_path, False)]
    assert len(splits.train_y) == 500 and len(splits.test_y) == 100


@pytest.mark.parametrize("val_size", [0, 600, 601])
def test_invalid_image_split_sizes(val_size):
    with pytest.raises(ValueError, match="val_size"):
        data.split_images(*data.demo_images(), val_size=val_size)


@pytest.mark.parametrize("stats", [(0, 0), (0, -1), (float("nan"), 1), (0, float("inf"))])
def test_invalid_normalization_statistics(stats):
    with pytest.raises(ValueError, match="statistics"):
        data.get_image_splits(demo=True, stats=stats)


def test_demo_training_and_evaluation_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_fashion_mnist", lambda *args: pytest.fail("demo attempted download"))
    checkpoint_path = tmp_path / "dnn.pt"
    output_path = tmp_path / "evaluation.json"
    train.main([
        "--demo", "--epochs", "2", "--hidden-sizes", "32", "16", "--dropout", "0",
        "--split-seed", "17", "--val-size", "120", "--device", "cpu", "--out", str(checkpoint_path),
    ])
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    evaluate.main(["--checkpoint", str(checkpoint_path), "--batch-size", "17", "--out", str(output_path)])
    result = json.loads(output_path.read_text())
    assert result["model_type"] == "dnn" and result["demo"]
    assert result["data_config"]["split_seed"] == 17
    assert result["pixel_mean"] == checkpoint["pixel_mean"]
    assert result["splits"]["val"]["examples"] == 120
    assert result["splits"]["test"]["examples"] == 100
    for split in result["splits"].values():
        assert 0 <= split["accuracy"] <= 1 and 0 <= split["macro_f1"] <= 1
        matrix = np.asarray(split["confusion_matrix"])
        assert matrix.shape == (10, 10) and matrix.sum() == split["examples"]
        assert split["accuracy"] == matrix.trace() / matrix.sum()
    assert result["splits"]["val"]["accuracy"] == checkpoint["val_accuracy"]


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
