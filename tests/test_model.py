import torch
from torch.nn import functional as F

from model import AdaptiveRecurrentClassifier


def test_shapes_and_shared_gradients():
    model = AdaptiveRecurrentClassifier(channels=3, samples=30, hidden_size=12, max_steps=4)
    x = torch.randn(6, 3, 30)
    out = model(x)
    assert out.shape == (6, 4, 2)
    loss = sum(F.cross_entropy(out[:, i], torch.randint(0, 2, (6,))) for i in range(4))
    loss.backward()
    assert model.recurrent.weight.grad is not None
    assert torch.isfinite(model.recurrent.weight.grad).all()


def test_fixed_equals_last_pass():
    model = AdaptiveRecurrentClassifier(samples=10, max_steps=4).eval()
    x = torch.randn(5, 3, 10)
    with torch.inference_mode():
        all_logits = model(x)
        for k in range(1, 5):
            assert torch.allclose(model.forward_fixed(x, k), all_logits[:, k - 1], atol=1e-6)


def test_forced_full_depth_equals_fixed():
    model = AdaptiveRecurrentClassifier(samples=10, max_steps=4).eval()
    x = torch.randn(8, 3, 10)
    with torch.inference_mode():
        logits, steps = model.predict_adaptive(x, confidence=1.0, stability=0.0)
        assert torch.all(steps == 4)
        assert torch.allclose(logits, model.forward_fixed(x), atol=1e-6)


def test_exit_predictions_match_selected_depth():
    model = AdaptiveRecurrentClassifier(samples=10, max_steps=4).eval()
    x = torch.randn(16, 3, 10)
    with torch.inference_mode():
        all_logits = model(x)
        chosen, steps = model.predict_adaptive(x, confidence=0.0, stability=1.0)
        for i in range(len(x)):
            assert torch.allclose(chosen[i], all_logits[i, steps[i] - 1], atol=1e-6)


def test_loose_thresholds_do_not_exit_before_min_steps():
    model = AdaptiveRecurrentClassifier(samples=10, max_steps=4).eval()
    x = torch.randn(5, 3, 10)
    with torch.inference_mode():
        logits, steps = model.predict_adaptive(x, confidence=0.0, stability=1.0)
        assert logits.shape == (5, 2)
        assert torch.all((2 <= steps) & (steps <= 4))
