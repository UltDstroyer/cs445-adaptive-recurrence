import pytest
import torch
from torch import nn
from torch.nn import functional as F

from model import MultiExitClassifier, MultiExitCNN, build_model


@pytest.fixture(params=["mlp", "cnn"])
def model(request):
    torch.manual_seed(0)
    return (MultiExitClassifier() if request.param == "mlp" else MultiExitCNN()).eval()


def test_mlp_design_matches_the_original_plan():
    model = MultiExitClassifier()
    assert model.projection.in_features == 784 and model.projection.out_features == 32
    for block, head in zip(model.blocks, model.heads):
        assert (block.expand.in_features, block.expand.out_features) == (32, 128)
        assert (block.contract.in_features, block.contract.out_features) == (128, 32)
        assert (head.in_features, head.out_features) == (32, 10)
    linear_weights = sum(m.weight.numel() + m.bias.numel() for m in model.modules() if isinstance(m, nn.Linear))
    assert linear_weights == 111_940
    assert sum(p.numel() for p in model.parameters()) == 112_580  # plus 10 LayerNorms


def test_cnn_design_matches_the_revised_plan():
    model = MultiExitCNN()
    assert sum(p.numel() for p in model.parameters()) == 71_668
    shapes = []
    h = model.stem(torch.rand(1, 1, 28, 28))
    for block in model.blocks:
        h = block(h)
        shapes.append(tuple(h.shape[1:]))
    assert shapes == [(16, 14, 14)] * 3 + [(32, 7, 7)] * 7


def test_output_is_batch_by_exit_by_class(model):
    assert model(torch.rand(7, 1, 28, 28)).shape == (7, 10, 10)


def test_no_blocks_or_heads_share_weights(model):
    pointers = [p.data_ptr() for p in model.parameters()]
    assert len(pointers) == len(set(pointers))
    assert len(model.blocks) == len(model.heads) == 10
    first_weights = [next(part.parameters()) for part in [*model.blocks, *model.heads]]
    assert len({w.data_ptr() for w in first_weights}) == 20


def test_every_exit_trains_every_block_and_head(model):
    logits = model.train()(torch.rand(12, 1, 28, 28))
    labels = torch.arange(12) % 10
    F.cross_entropy(logits.flatten(0, 1), labels.repeat_interleave(10)).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_exit_k_only_uses_the_first_k_blocks(model):
    x = torch.rand(4, 1, 28, 28)
    before = model(x)
    with torch.no_grad():
        for parameter in model.blocks[5].parameters():
            parameter.add_(0.5)
    after = model(x)
    torch.testing.assert_close(after[:, :5], before[:, :5])
    assert not torch.allclose(after[:, 5:], before[:, 5:])


def test_never_stopping_reproduces_the_last_exit(model):
    x = torch.rand(9, 1, 28, 28)
    logits, exits = model.predict_sequential(x, lambda number, history: torch.zeros(len(history), dtype=torch.bool))
    assert exits.tolist() == [10] * 9
    torch.testing.assert_close(logits, model(x)[:, -1])


@pytest.mark.parametrize("stop_at", [1, 2, 7, 10])
def test_stopping_at_a_fixed_exit_matches_that_exit(model, stop_at):
    x = torch.rand(6, 1, 28, 28)
    logits, exits = model.predict_sequential(
        x, lambda number, history: torch.full((len(history),), number == stop_at),
    )
    assert exits.tolist() == [stop_at] * 6
    torch.testing.assert_close(logits, model(x)[:, stop_at - 1])


def test_images_stop_independently_and_see_their_own_history(model):
    x = torch.rand(10, 1, 28, 28)
    targets = torch.tensor([1, 3, 10, 2, 5, 5, 9, 4, 6, 8])
    full = model(x)
    batch_sizes = []

    def stop_at_target(number, history):
        batch_sizes.append(len(history))
        # Identify each running image by its exit-1 logits, which differ between images.
        ids = torch.cdist(history[:, 0], full[:, 0]).argmin(dim=1)
        torch.testing.assert_close(history, full[ids, :number])
        return targets[ids] == number

    logits, exits = model.predict_sequential(x, stop_at_target)
    assert exits.tolist() == targets.tolist()
    torch.testing.assert_close(logits, full[torch.arange(10), targets - 1])
    # Stopped images leave the batch; the rule is not consulted at the last exit.
    assert batch_sizes == [int((targets >= n).sum()) for n in range(1, 10)]


def test_min_exit_prevents_earlier_stops(model):
    x = torch.rand(5, 1, 28, 28)
    calls = []

    def always(number, history):
        calls.append(number)
        return torch.ones(len(history), dtype=torch.bool)

    logits, exits = model.predict_sequential(x, always, min_exit=2)
    assert calls == [2] and exits.tolist() == [2] * 5
    torch.testing.assert_close(logits, model(x)[:, 1])


def test_invalid_inputs_are_rejected(model):
    with pytest.raises(ValueError, match="input shape"):
        model(torch.rand(2, 784))
    with pytest.raises(ValueError, match="min_exit"):
        model.predict_sequential(torch.rand(2, 1, 28, 28), lambda n, h: torch.ones(len(h)), min_exit=11)
    with pytest.raises(ValueError, match="should_stop must return"):
        model.predict_sequential(torch.rand(3, 1, 28, 28), lambda n, h: torch.ones(1, dtype=torch.bool))


def test_invalid_designs_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        MultiExitClassifier(num_exits=0)
    with pytest.raises(ValueError, match="downsample_at"):
        MultiExitCNN(downsample_at=11)
    with pytest.raises(ValueError, match="positive"):
        MultiExitCNN(channels=(0, 32))


def test_build_model_rebuilds_both_architectures_from_saved_configs():
    cnn = build_model({"architecture": "cnn", "input_shape": [1, 28, 28], "channels": [16, 32],
                       "downsample_at": 4, "num_exits": 10, "num_classes": 10})
    assert isinstance(cnn, MultiExitCNN)
    # MLP checkpoints saved before the CNN existed have no "architecture" key.
    mlp = build_model({"input_shape": [1, 28, 28], "width": 32, "block_hidden": 128,
                       "num_exits": 10, "num_classes": 10})
    assert isinstance(mlp, MultiExitClassifier)
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model({"architecture": "rnn"})
