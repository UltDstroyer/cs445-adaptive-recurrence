"""Compute accounting: measured multiply-adds must match the hand count in the plan."""

import numpy as np
import pytest
import torch

from exit_outputs import ComputeProfile, ExitOutputs
from model import MultiExitClassifier, MultiExitCNN, count_multiply_adds

# Hand count from IMPLEMENTATION_PLAN.md, section 1.1 (CNN):
#   stem       3x3 conv 1->16 at 14x14:            1*16*9*196 =  28,224
#   blocks 1-3 3x3 conv 16->16 at 14x14:          16*16*9*196 = 451,584
#   block 4    3x3 conv 16->32 at 7x7:             16*32*9*49 = 225,792
#              + 1x1 shortcut 16->32 at 7x7:          16*32*49 =  25,088
#   blocks 5-10 3x3 conv 32->32 at 7x7:            32*32*9*49 = 451,584
#   heads      linear 16->10 or 32->10:                          160 / 320
CNN_HAND_COUNT = {
    "stem": 28_224,
    "blocks": [451_584] * 3 + [250_880] + [451_584] * 6,
    "heads": [160] * 3 + [320] * 7,
}
CNN_SHARES = [11.0, 21.4, 31.8, 37.6, 48.0, 58.4, 68.8, 79.2, 89.6, 100.0]

# MLP (the first design): projection 784*32, blocks 2*32*128, heads 32*10.
MLP_HAND_COUNT = {"stem": 25_088, "blocks": [8_192] * 10, "heads": [320] * 10}


def test_cnn_count_matches_the_hand_count():
    counts = count_multiply_adds(MultiExitCNN())
    assert counts == CNN_HAND_COUNT
    profile = ComputeProfile.from_dict(counts)
    assert profile.full == 4_346_080
    assert profile.cumulative()[:2].tolist() == [479_968, 931_712]
    assert [round(100 * f, 1) for f in profile.fractions()] == CNN_SHARES


def test_mlp_count_matches_the_hand_count():
    profile = ComputeProfile.from_dict(count_multiply_adds(MultiExitClassifier()))
    assert profile == ComputeProfile.from_dict(MLP_HAND_COUNT)
    assert profile.full == 110_208
    assert round(100 * profile.fractions()[1], 1) == 38.2  # the MLP's cheapest allowed stop


def test_counting_does_not_change_the_model():
    model = MultiExitCNN().train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    count_multiply_adds(model)
    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_counts_scale_with_the_design():
    wider = count_multiply_adds(MultiExitCNN(channels=(32, 64)))
    assert wider["blocks"][0] == 32 * 32 * 9 * 196
    assert wider["heads"][-1] == 64 * 10


def test_fixed_exits_pay_only_for_their_own_head():
    profile = ComputeProfile.from_dict(CNN_HAND_COUNT)
    adaptive, fixed = profile.cumulative(all_heads=True), profile.cumulative(all_heads=False)
    assert adaptive[0] == fixed[0] == 28_224 + 451_584 + 160
    assert adaptive[9] - fixed[9] == sum(CNN_HAND_COUNT["heads"][:9])
    assert profile.fractions(all_heads=True)[-1] == 1.0
    assert profile.fractions(all_heads=False)[-1] < 1.0


def test_mean_fraction_averages_over_images():
    profile = ComputeProfile.from_dict(CNN_HAND_COUNT)
    fractions = profile.fractions()
    assert profile.mean_fraction([10, 10]) == 1.0
    assert profile.mean_fraction([2, 4, 4, 10]) == pytest.approx((fractions[1] + 2 * fractions[3] + 1) / 4)
    for bad in ([], [0], [11]):
        with pytest.raises(ValueError, match="exit numbers"):
            profile.mean_fraction(bad)


def test_profiles_are_read_from_saved_outputs():
    outputs = ExitOutputs(
        logits=np.zeros((2, 10, 10), dtype=np.float32), labels=np.array([0, 1]),
        image_indices=np.array([0, 1]), metadata={"multiply_adds": CNN_HAND_COUNT},
    )
    assert outputs.compute_profile().full == 4_346_080
    with pytest.raises(ValueError, match="no compute profile"):
        ExitOutputs(outputs.logits, outputs.labels, outputs.image_indices, {}).compute_profile()
    with pytest.raises(ValueError, match="number of exits"):
        ExitOutputs(np.zeros((2, 3, 10), dtype=np.float32), outputs.labels, outputs.image_indices,
                    {"multiply_adds": CNN_HAND_COUNT}).compute_profile()


def test_invalid_profiles_are_rejected():
    with pytest.raises(ValueError, match="one block and one head"):
        ComputeProfile(10, (1, 2), (1,))
    with pytest.raises(ValueError, match="negative"):
        ComputeProfile(-1, (1,), (1,))
