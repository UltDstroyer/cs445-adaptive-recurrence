import pytest
import torch
from torch.nn import functional as F

from model import AdaptiveRecurrentClassifier, model_from_checkpoint


@pytest.mark.parametrize("encoder_type", ["dense", "eeg"])
def test_shapes_and_shared_gradients(encoder_type):
    model = AdaptiveRecurrentClassifier(channels=3, samples=30, hidden_size=12, max_steps=4, encoder_type=encoder_type)
    x = torch.randn(6, 3, 30)
    out = model(x)
    assert out.shape == (6, 4, 2)
    loss = sum(F.cross_entropy(out[:, i], torch.randint(0, 2, (6,))) for i in range(4))
    loss.backward()
    assert model.recurrent.weight.grad is not None
    assert torch.isfinite(model.recurrent.weight.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.encoder.parameters())


@pytest.mark.parametrize("encoder_type", ["dense", "eeg"])
def test_fixed_equals_last_pass(encoder_type):
    model = AdaptiveRecurrentClassifier(samples=10, max_steps=4, encoder_type=encoder_type).eval()
    x = torch.randn(5, 3, 10)
    with torch.inference_mode():
        all_logits = model(x)
        for k in range(1, 5):
            assert torch.allclose(model.forward_fixed(x, k), all_logits[:, k - 1], atol=1e-6)


@pytest.mark.parametrize("encoder_type", ["dense", "eeg"])
def test_forced_full_depth_equals_fixed(encoder_type):
    model = AdaptiveRecurrentClassifier(samples=10, max_steps=4, encoder_type=encoder_type).eval()
    x = torch.randn(8, 3, 10)
    with torch.inference_mode():
        logits, steps = model.predict_adaptive(x, confidence=1.0, stability=0.0)
        assert torch.all(steps == 4)
        assert torch.allclose(logits, model.forward_fixed(x), atol=1e-6)


@pytest.mark.parametrize("encoder_type", ["dense", "eeg"])
def test_exit_predictions_match_selected_depth(encoder_type):
    model = AdaptiveRecurrentClassifier(samples=10, max_steps=4, encoder_type=encoder_type).eval()
    x = torch.randn(16, 3, 10)
    with torch.inference_mode():
        all_logits = model(x)
        chosen, steps = model.predict_adaptive(x, confidence=0.0, stability=1.0)
        for i in range(len(x)):
            assert torch.allclose(chosen[i], all_logits[i, steps[i] - 1], atol=1e-6)


@pytest.mark.parametrize("encoder_type", ["dense", "eeg"])
def test_loose_thresholds_do_not_exit_before_min_steps(encoder_type):
    model = AdaptiveRecurrentClassifier(samples=10, max_steps=4, encoder_type=encoder_type).eval()
    x = torch.randn(5, 3, 10)
    with torch.inference_mode():
        logits, steps = model.predict_adaptive(x, confidence=0.0, stability=1.0)
        assert logits.shape == (5, 2)
        assert torch.all((2 <= steps) & (steps <= 4))


def test_dropout_is_disabled_during_inference():
    model = AdaptiveRecurrentClassifier(samples=100, dropout=0.5)
    x = torch.randn(8, 3, 100)
    model.train()
    assert not torch.allclose(model(x), model(x))
    model.eval()
    with torch.inference_mode():
        assert torch.equal(model(x), model(x))


def test_legacy_dense_checkpoint_predictions_are_preserved():
    config = dict(channels=3, samples=10, hidden_size=12, num_classes=2, max_steps=4)
    original = AdaptiveRecurrentClassifier(**config, encoder_type="dense", dropout=0).eval()
    checkpoint = {"model_config": config, "model_state": original.state_dict()}
    restored = model_from_checkpoint(checkpoint).eval()
    assert restored.encoder_type == "dense" and restored.dropout == 0
    assert checkpoint["model_config"] == config
    with torch.inference_mode():
        x = torch.randn(5, 3, 10)
        assert torch.equal(original(x), restored(x))


def test_eeg_checkpoint_round_trip_preserves_predictions(tmp_path):
    config = dict(channels=3, samples=100, hidden_size=12, num_classes=2, max_steps=3, encoder_type="eeg", dropout=0.25)
    original = AdaptiveRecurrentClassifier(**config).eval()
    path = tmp_path / "model.pt"
    torch.save({"model_config": config, "model_state": original.state_dict()}, path)
    restored = model_from_checkpoint(torch.load(path, weights_only=True)).eval()
    with torch.inference_mode():
        x = torch.randn(5, 3, 100)
        assert torch.equal(original(x), restored(x))
