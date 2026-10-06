import pytest
from agi.training_progress import StepInterval, validate_batches


def test_schedule_requires_actual_learning_and_handles_overshoot_and_resume():
    interval = StepInterval(1000)
    assert not interval.due(0)
    assert not interval.due(0)
    assert not interval.due(999)
    assert interval.due(1200)
    assert not interval.due(1200)
    resumed = StepInterval(1000, previous=1200)
    assert not resumed.due(2000)
    assert resumed.due(2300)
    assert not StepInterval(0).due(100000)


@pytest.mark.parametrize('size', [0, 1, 128, 225])
def test_invalid_impala_batch_fails_before_training(size):
    with pytest.raises(ValueError, match='divisible'):
        validate_batches({'train_batch_size': size, 'rollout_fragment_length': 50})


def test_valid_impala_batch():
    validate_batches({'train_batch_size': 1000, 'minibatch_size': 200, 'rollout_fragment_length': 50})


def test_uniform_opponents_keep_easy_historical_policies_in_training_mix():
    import numpy as np
    from agi.league import opponent_probabilities
    rewards = [-10., -5., 0., 5., 10., 15.]
    ranked = opponent_probabilities(rewards, 'ranked', k=2)
    uniform = opponent_probabilities(rewards, 'uniform', k=2)
    assert ranked[0] > ranked[-1]
    np.testing.assert_allclose(uniform, np.full(6, 1 / 6))
    assert uniform[-1] > ranked[-1]
    assert uniform.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        opponent_probabilities([], 'uniform')
