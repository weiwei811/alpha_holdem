import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from agi.nl_holdem_env import NlHoldemEnvWrapper, NlHoldemEnvWithOpponent
from agi.nl_holdem_structured_net import NlHoldemStructuredNet, actor_relative_observation


def config():
    return ast.literal_eval(Path('confs/nl_holdem_structured.py').read_text())


def tensors(obs):
    return {k:torch.as_tensor(v).unsqueeze(0) for k,v in obs.items()}


@pytest.mark.parametrize('players',[2,6])
def test_actor_relative_model_is_invariant_to_cyclic_seat_relabeling(players):
    conf=config();conf['env_config']['custom_options']['num_players']=players
    env=NlHoldemEnvWrapper(conf);obs,_=env.reset(seed=5)
    batch=tensors(obs);rotated={k:v.clone() for k,v in batch.items()}
    for key in ('table_info','extra_info'):
        rotated[key]=torch.roll(rotated[key],1,dims=1)
    rotated['action_info'][:,:players]=torch.roll(rotated['action_info'][:,:players],1,dims=1)
    # Exercise history rows as well as actor/button flags.
    batch['action_info'][0,0,0,0]=1
    rotated['action_info'][0,1,0,0]=1
    left=actor_relative_observation(batch);right=actor_relative_observation(rotated)
    for key in left:torch.testing.assert_close(left[key],right[key])
    model=NlHoldemStructuredNet(env.observation_space,env.action_space,5,conf['model'],'test')
    a,_=model({'obs':batch},[],None);va=model.value_function().clone()
    b,_=model({'obs':rotated},[],None);vb=model.value_function()
    torch.testing.assert_close(a,b);torch.testing.assert_close(va,vb)
    assert a.shape==(1,5) and va.shape==(1,)
    assert obs['legal_moves'][a.argmax().item()]


def test_public_betting_features_use_street_ledger_and_actual_short_call():
    env=NlHoldemEnvWrapper(config());obs,_=env.reset(seed=17)
    assert env.observation_space.contains(obs)
    actor=env.my_agent();game=env.env.game
    game.players[actor].remained_chips=10
    game.round.raised=[0]*6;game.round.raised[(actor+1)%6]=50
    # A previously short all-in has a different total contribution.
    game.players[(actor+2)%6].in_chips=400
    raw=game.get_state(actor)
    obs=env._get_observation(({'raw_obs':raw},actor))
    info=obs['betting_info']
    assert info[0]==sum(p.in_chips for p in game.players)
    assert info[1]==50 and info[2]==10
    assert info[6]==pytest.approx(10/(info[0]+10))
    assert info[4]==10
    assert np.isfinite(info).all()
    assert env.observation_space.contains(obs)
    # Opponents' private cards must not affect features.
    game.players[(actor+1)%6].hand=list(game.players[(actor+2)%6].hand)
    other=env._get_observation(({'raw_obs':game.get_state(actor)},actor))
    for key in obs:np.testing.assert_array_equal(obs[key],other[key])


def test_critic_gradients_do_not_modify_policy_encoder():
    conf=config();env=NlHoldemEnvWrapper(conf);obs,_=env.reset(seed=42)
    model=NlHoldemStructuredNet(env.observation_space,env.action_space,5,conf['model'],'test')
    batch=tensors(obs);model({'obs':batch},[],None)
    model.value_function().sum().backward()
    assert all(p.grad is None for p in model.policy_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.value_encoder.parameters())
    model.zero_grad(set_to_none=True)
    logits,_=model({'obs':batch},[],None)
    logits[:,obs['legal_moves'].astype(bool)].sum().backward()
    assert all(p.grad is None for p in model.value_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.policy_encoder.parameters())


def test_shared_critic_ablation_and_checkpoint_roundtrip(tmp_path):
    import pickle
    from agi.evaluation_tools import NNAgent
    conf=config();conf['model']['custom_model_config']['separate_critic']=False
    env=NlHoldemEnvWrapper(conf);obs,_=env.reset(seed=3)
    agent=NNAgent(env.observation_space,env.action_space,conf,None,seed=3)
    logits,_=agent.model({'obs':tensors(obs)},[],None)
    assert torch.isfinite(logits).all() and torch.isfinite(agent.model.value_function()).all()
    path=tmp_path/'weights.pkl'
    with path.open('wb') as f:pickle.dump({k:v.detach().numpy() for k,v in agent.model.state_dict().items()},f)
    loaded=NNAgent(env.observation_space,env.action_space,conf,path)
    assert agent.make_action(obs,deterministic=True)==loaded.make_action(obs,deterministic=True)


def test_scaled_terminal_reward_is_raw_payoff_for_real_multiway_hands():
    conf=config();env=NlHoldemEnvWithOpponent(conf,opponent='random')
    rng=np.random.default_rng(10)
    for seed in range(20):
        obs,_=env.reset(seed=seed);done=False;total=0.
        while not done:
            action=int(rng.choice(np.flatnonzero(obs['legal_moves'])))
            obs,reward,done,truncated,_=env.step(action)
            assert not truncated
            if not done:assert reward==0
            total+=reward
        assert total*env.reward_scale==pytest.approx(env.last_reward)


def test_vtrace_terminal_boundaries_do_not_leak_bootstrap_to_previous_hands():
    from ray.rllib.algorithms.impala.vtrace_torch import from_importance_weights
    rewards=torch.tensor([[0.],[1.],[0.],[-.5]])
    discounts=torch.tensor([[1.],[0.],[1.],[0.]])
    result=from_importance_weights(log_rhos=torch.zeros_like(rewards),discounts=discounts,
                                  rewards=rewards,values=torch.full_like(rewards,.3),
                                  bootstrap_value=torch.tensor([100.]))
    torch.testing.assert_close(result.vs,torch.tensor([[1.],[1.],[-.5],[-.5]]))
    # A genuinely cut final fragment must keep its next-observation bootstrap.
    discounts[-1]=1
    result=from_importance_weights(log_rhos=torch.zeros_like(rewards),discounts=discounts,
                                  rewards=rewards,values=torch.full_like(rewards,.3),
                                  bootstrap_value=torch.tensor([2.]))
    torch.testing.assert_close(result.vs,torch.tensor([[1.],[1.],[1.5],[1.5]]))


def test_checkpoint_config_rejects_changed_seat_semantics_before_ray(tmp_path):
    import json
    from agi.training_progress import validate_restore_configuration
    conf=config()
    (tmp_path/'training_config.json').write_text(json.dumps(conf))
    validate_restore_configuration(conf,tmp_path)
    conf['model']['custom_model_config']['relative_seats']=False
    with pytest.raises(ValueError,match='differs'):
        validate_restore_configuration(conf,tmp_path)


def test_structured_adapter_requires_explicit_public_street_state():
    from agi.evaluation_tools import NNAgent
    conf=config();env=NlHoldemEnvWrapper(conf);obs,_=env.reset(seed=9)
    raw=env.last_obs[0]['raw_obs'];game=env.env.game
    board=dict(raw,legal_moves=[a.value for a in raw['legal_actions']],
               contributions=[p.in_chips for p in game.players],
               statuses=['alive']*6,dealer_id=game.dealer_id)
    agent=NNAgent(env.observation_space,env.action_space,conf,None)
    with pytest.raises(ValueError,match='street_contributions'):
        agent.model_deci([[],[],[],[]],board)
    board.update(street_contributions=list(game.round.raised),last_raise_amount=game.round.last_raise_amount)
    expected=agent.make_action(obs,deterministic=True)
    assert agent.model_deci([[],[],[],[]],board,deterministic=True)==expected


@pytest.mark.parametrize('terminal',[True,False])
def test_real_impala_postprocessor_uses_zero_terminal_and_next_obs_for_cut(terminal):
    from ray.rllib.models import ModelCatalog
    from ray.rllib.algorithms.impala import ImpalaConfig
    from ray.rllib.policy.sample_batch import SampleBatch
    from agi.portable_impala import PortableImpalaTorchPolicy
    conf=config();env=NlHoldemEnvWithOpponent(conf,opponent='random')
    ModelCatalog.register_custom_model('NlHoldemStructuredNet',NlHoldemStructuredNet)
    prep=ModelCatalog.get_preprocessor_for_space(env.observation_space)
    algo=(ImpalaConfig().framework('torch').resources(num_gpus=0)
          .api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False))
    algo.model.update(conf['model']);algo.env_config=conf['env_config']
    policy=PortableImpalaTorchPolicy(prep.observation_space,env.action_space,algo.to_dict())
    for seed in range(100):
        obs,_=env.reset(seed=seed)
        next_obs,reward,done,_,_=env.step(0 if terminal else 1)
        if done==terminal:break
    else:raise AssertionError('Could not construct trajectory boundary')
    encoded=prep.transform(obs);next_encoded=prep.transform(next_obs)
    _,_,extra=policy.compute_actions(np.asarray([encoded]))
    _,_,next_extra=policy.compute_actions(np.asarray([next_encoded]))
    batch=SampleBatch({'obs':np.asarray([encoded]),'new_obs':np.asarray([next_encoded]),
                      'actions':np.asarray([0 if terminal else 1]),'rewards':np.asarray([reward],dtype=np.float32),
                      'terminateds':np.asarray([terminal]),'truncateds':np.asarray([False]),
                      'vf_preds':extra['vf_preds']})
    result=policy.postprocess_trajectory(batch)
    expected=0 if terminal else next_extra['vf_preds'][0]
    assert result[SampleBatch.VALUES_BOOTSTRAPPED][-1]==pytest.approx(expected)


def test_neural_opponent_accepts_minimal_structured_model_config():
    from ray.rllib.models import ModelCatalog
    ModelCatalog.register_custom_model('NlHoldemStructuredNet', NlHoldemStructuredNet)
    env = NlHoldemEnvWithOpponent(config())
    obs, _ = env.reset(seed=42)
    assert env.oppo_policy.config['model']['max_seq_len'] > 0
    for _ in range(100):
        action = int(np.flatnonzero(obs['legal_moves'])[0])
        obs, reward, done, truncated, _ = env.step(action)
        assert np.isfinite(reward) and not truncated
        if done:
            break
    else:
        raise AssertionError('Neural-opponent hand did not finish')


def test_explicit_entropy_override_survives_restored_policy_config():
    from ray.rllib.models import ModelCatalog
    from ray.rllib.algorithms.impala import ImpalaConfig
    from agi.portable_impala import PortableImpalaTorchPolicy, configure_entropy
    conf=config();env=NlHoldemEnvWrapper(conf)
    ModelCatalog.register_custom_model('NlHoldemStructuredNet',NlHoldemStructuredNet)
    prep=ModelCatalog.get_preprocessor_for_space(env.observation_space)
    algo=ImpalaConfig().framework('torch').resources(num_gpus=0).api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)
    algo.model.update(conf['model']);algo.env_config=conf['env_config'];algo.entropy_coeff=.1
    policy=PortableImpalaTorchPolicy(prep.observation_space,env.action_space,algo.to_dict())
    saved=policy.get_state();weights=policy.get_weights()
    policy.set_state(saved);configure_entropy(policy,.02)
    policy.on_global_var_update({'timestep':2000000})
    assert policy.entropy_coeff==pytest.approx(.02)
    assert policy.config['entropy_coeff']==.02
    for k,v in policy.get_weights().items():np.testing.assert_array_equal(v,weights[k])
    with pytest.raises(ValueError):configure_entropy(policy,float('nan'))
