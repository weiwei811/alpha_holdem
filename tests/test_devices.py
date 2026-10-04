import numpy as np
import pytest
import torch
from agi.devices import resolve_device, configure_training_device


@pytest.mark.parametrize('cuda,mps,expected', [(True,False,'cuda'),(False,True,'mps'),(False,False,'cpu')])
def test_auto_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: mps)
    assert resolve_device().type == expected


def test_explicit_unavailable_device_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    for backend in ('cuda','mps'):
        with pytest.raises(ValueError, match='unavailable'):
            resolve_device(backend)


def test_cpu_override_and_mps_resource_configuration(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    conf = {}
    assert configure_training_device(conf, 'auto', gpus=0).type == 'cpu'
    assert conf['num_gpus'] == 0
    conf = {}
    assert configure_training_device(conf, 'mps').type == 'mps'
    assert conf['num_gpus'] == 0 and conf['simple_optimizer']
    assert conf['env_config']['learner_device'] == 'mps'
    with pytest.raises(ValueError, match='conflicts'):
        configure_training_device({}, 'cuda', gpus=0)


@pytest.mark.parametrize('device', ['cpu','cuda','mps'])
def test_inference_and_backward_on_available_device(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA hardware/build unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS hardware/build unavailable')
    from agi.nl_holdem_env import NlHoldemEnvWrapper
    from agi.evaluation_tools import NNAgent
    conf = {'env_config': {'custom_options': {'num_players': 6}}, 'model': {'custom_model': 'NlHoldemNet'}}
    env = NlHoldemEnvWrapper(conf)
    obs,_ = env.reset(seed=5)
    agent = NNAgent(env.observation_space, env.action_space, conf, None, device=device)
    assert next(agent.model.parameters()).device.type == device
    assert obs['legal_moves'][agent.make_action(obs, deterministic=True)]
    tensors = {k:torch.as_tensor(v,device=device).unsqueeze(0) for k,v in obs.items()}
    logits,_ = agent.model({'obs': tensors}, [], None)
    legal = torch.as_tensor(obs['legal_moves'],device=device,dtype=torch.bool)
    loss = logits[:,legal].sum() + agent.model.value_function().sum()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in agent.model.parameters() if p.grad is not None)


@pytest.mark.parametrize('device', ['cpu','cuda','mps'])
def test_portable_single_device_adapter_updates_real_impala_policy(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA hardware/build unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS hardware/build unavailable')
    from ray.rllib.algorithms.impala import ImpalaConfig
    from ray.rllib.models import ModelCatalog
    from agi.nl_holdem_env import NlHoldemEnvWrapper
    from agi.nl_holdem_net import NlHoldemNet
    from agi.portable_impala import PortableImpalaTorchPolicy, move_policy_to_device
    ModelCatalog.register_custom_model('NlHoldemNet', NlHoldemNet)
    conf = {'env_config': {'custom_options': {'num_players': 6}}, 'model': {'custom_model': 'NlHoldemNet'}}
    env = NlHoldemEnvWrapper(conf)
    obs,_ = env.reset(seed=5)
    prep = ModelCatalog.get_preprocessor_for_space(env.observation_space)
    config = (ImpalaConfig().framework('torch').resources(num_gpus=0)
              .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False))
    config.model['custom_model'] = 'NlHoldemNet'
    config.rollout_fragment_length = 4
    policy = PortableImpalaTorchPolicy(prep.observation_space,env.action_space,config.to_dict())
    move_policy_to_device(policy, torch.device(device))
    # Exercise the MPS-compatible gradient path on actual IMPALA/V-trace loss.
    policy._metal_learner = True
    batch = policy._dummy_batch.copy()
    batch.set_get_interceptor(None)
    encoded = np.stack([prep.transform(obs)] * batch.count)
    actions, _, extra = policy.compute_actions(encoded)
    batch['obs'] = encoded
    batch['new_obs'] = encoded.copy()
    batch['actions'] = actions
    batch['action_logp'] = extra['action_logp']
    batch['action_dist_inputs'] = extra['action_dist_inputs']
    batch['rewards'] = np.ones(batch.count, dtype=np.float32)
    batch['terminateds'] = np.ones(batch.count, dtype=bool)
    batch['truncateds'] = np.zeros(batch.count, dtype=bool)
    batch = policy.postprocess_trajectory(batch)
    before = {k:v.copy() for k,v in policy.get_weights().items()}
    policy.learn_on_batch(batch)
    after = policy.get_weights()
    assert all(np.isfinite(v).all() for v in after.values())
    assert any(not np.array_equal(before[k],v) for k,v in after.items())
