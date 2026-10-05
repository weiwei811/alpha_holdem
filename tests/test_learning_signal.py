"""Catch input-dependent signal suppression before running long training jobs."""
import numpy as np
import pytest
import torch
from agi.nl_holdem_env import NlHoldemEnvWrapper
from agi.nl_holdem_net import NlHoldemNet
from agi.nl_holdem_lg_net import NlHoldemLgNet


@pytest.mark.parametrize("model_class", [NlHoldemNet, NlHoldemLgNet])
def test_fresh_policy_distinguishes_cards_and_backpropagates_to_features(model_class):
    torch.manual_seed(42)
    env = NlHoldemEnvWrapper({"env_config": {"custom_options": {"num_players": 6}}})
    observations = [env.reset(seed=seed)[0] for seed in range(32)]
    batch = {key: torch.tensor(np.stack([obs[key] for obs in observations])) for key in observations[0]}
    # Hold public information and masks fixed, varying only the hero's private cards.
    for key in batch:
        if key != "card_info":
            batch[key] = batch[key][0:1].repeat(32, *([1] * (batch[key].ndim - 1)))
    batch["legal_moves"] = torch.ones(32, 5)
    model = model_class(env.observation_space, env.action_space, 5, {}, "test")
    logits, _ = model({"obs": batch}, [], None)
    assert logits.softmax(-1).std(0).max().item() > 1e-6
    assert model.value_function().std().item() > 1e-5
    targets = batch["card_info"][:, :, 12, 0].sum(1).clamp(max=1).long()
    loss = torch.nn.functional.cross_entropy(logits[:, :2], targets)
    loss.backward()
    assert model.card_conv[0].conv1.weight.grad.norm().item() > 1e-7
