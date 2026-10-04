import pickle
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("has_final", [False, True])
def test_restore_prefers_final_learner_and_keeps_historical_pool(tmp_path, monkeypatch, has_final):
    import ray
    from utils import get_winrate_and_weight

    historical = {"weight": np.array([1.0])}
    final = {"weight": np.array([9.0])}
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "c_0.pkl").write_bytes(pickle.dumps(historical))
    (tmp_path / "winrates.csv").write_text("oppo,self-play,c_0\nwinrate,0,2\nmatches,0,10\n")
    if has_final:
        (tmp_path / "output_weight.pkl").write_bytes(pickle.dumps(final))
    pool, rewards = [], []
    league = SimpleNamespace(
        add_weight=SimpleNamespace(remote=lambda weight: pool.append(weight)),
        set_winrates=SimpleNamespace(remote=lambda values: rewards.extend(values)))
    monkeypatch.setattr(ray, "get", lambda value: value)
    restored = get_winrate_and_weight(tmp_path, league)
    assert restored["weight"].tolist() == ([9.0] if has_final else [1.0])
    assert len(pool) == 1 and pool[0]["weight"].tolist() == [1.0]
    assert rewards == [2.0]


def test_blind_walk_counts_once_and_opponent_prepared_before_actions():
    from agi.nl_holdem_env import NlHoldemEnvWithOpponent, NlHoldemEnvWrapper

    env = NlHoldemEnvWithOpponent(
        {"env_config": {"custom_options": {"num_players": 6}}}, opponent="random")
    env.env.game.configured_dealer_id = 0
    seats = iter([2, 3])  # BB wins without acting, then UTG gets a decision.
    env._np_random = SimpleNamespace(integers=lambda count: next(seats))
    records, events = [], []
    env.prepare_opponent = lambda current: events.append("prepared")
    env.on_hand_end = lambda current, reward: records.append((current.our_pid, reward))

    def folding_opponents(obs):
        assert events == ["prepared"]
        reward, done = [0] * 6, False
        while env.my_agent() != env.our_pid:
            obs, reward, done, info = NlHoldemEnvWrapper._inner_step(env, 0)
            if done:
                break
        return obs, reward, done, {}

    env._opponent_step = folding_opponents
    obs, _ = env.reset()
    assert records == [(2, 1.0)]
    assert env.our_pid == env.my_agent() == 3
    assert not env.is_done and env.observation_space.contains(obs)
    _, reward, done, _, _ = env.step(0)
    assert done and env.is_done
    assert len(records) == 2  # The blind walk and the decision hand each count once.
    assert records[-1] == (3, 0.0)
    assert reward == 0.0


def test_latest_learner_save_replaces_complete_snapshot(tmp_path):
    from utils import save_learner_weights
    save_learner_weights(tmp_path, {"weight": np.array([1.0])})
    save_learner_weights(tmp_path, {"weight": np.array([7.0])})
    with (tmp_path / "output_weight.pkl").open("rb") as stream:
        assert pickle.load(stream)["weight"].tolist() == [7.0]
    assert not (tmp_path / "output_weight.pkl.tmp").exists()
