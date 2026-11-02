import numpy as np

from data import demo_trials, split_trials


def test_demo_split_and_training_only_normalization():
    x, y, session = demo_trials()
    splits = split_trials(x, y, session)
    assert splits.train_x.shape == (144, 3, 301)
    assert splits.val_x.shape == splits.test_x.shape == (48, 3, 301)
    assert len(splits.session_names) == 5
    np.testing.assert_allclose(splits.train_x.mean(axis=(0, 2)), 0, atol=1e-5)
    np.testing.assert_allclose(splits.train_x.std(axis=(0, 2)), 1, atol=1e-5)


def test_supplied_stats_are_used_on_validation_and_test():
    x, y, session = demo_trials()
    reference = split_trials(x, y, session)
    other = split_trials(x, y, session, (reference.mean.squeeze(), reference.std.squeeze()))
    np.testing.assert_array_equal(reference.test_x, other.test_x)
