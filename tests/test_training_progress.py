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
