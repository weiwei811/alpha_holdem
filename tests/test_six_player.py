import numpy as np
import pytest
import rlcard
from rlcard.games.limitholdem import PlayerStatus
from rlcard.games.nolimitholdem import Action
from rlcard.games.base import Card


def test_all_in_seat_is_never_asked_to_act_again():
    env = rlcard.make('no-limit-holdem', config={'game_num_players': 6, 'dealer_id': 0})
    env.reset()
    env.game.players[env.get_player_id()].remained_chips = 50
    env.step(Action.ALL_IN.value)
    for _ in range(4):
        env.step(Action.CHECK_CALL.value)
    env.step(Action.RAISE_HALF_POT.value)
    assert env.game.players[env.get_player_id()].status == PlayerStatus.ALIVE



def test_side_pot_uses_best_eligible_hand():
    env = rlcard.make('no-limit-holdem', config={'game_num_players': 6})
    env.reset()
    board = [Card(s, r) for s, r in [('S','2'),('H','3'),('D','7'),('C','9'),('S','J')]]
    for i, player in enumerate(env.game.players):
        player.in_chips = [50, 100, 100, 0, 0, 0][i]
        player.status = PlayerStatus.ALLIN if i < 3 else PlayerStatus.FOLDED
        if i < 3:
            rank = ['A','K','Q'][i]
            player.hand = [Card('H', rank), Card('D', rank)]
    env.game.public_cards = board
    assert env.get_payoffs().tolist() == [100, 0, -100, 0, 0, 0]


def test_observation_preserves_large_stacks_and_contains_fold():
    from agi.nl_holdem_env import NlHoldemEnvWrapper
    env = NlHoldemEnvWrapper({'env_config': {'custom_options': {'num_players': 6}}})
    obs, _ = env.reset(seed=17)
    actor = env.my_agent()
    env.env.game.players[(actor + 1) % 6].remained_chips = 300
    obs, _, done, _, _ = env.step(0)
    assert not done
    assert obs['action_info'][actor, 0, 0] == 1
    assert obs['extra_info'][(actor + 1) % 6] == 300
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize('players,slots', [(2,6),(6,20),(6,30)])
@pytest.mark.parametrize('large', [False,True])
def test_network_matches_configured_observation(players, slots, large):
    import torch
    from agi.nl_holdem_env import NlHoldemEnvWrapper
    from agi.nl_holdem_net import NlHoldemNet
    from agi.nl_holdem_lg_net import NlHoldemLgNet
    env = NlHoldemEnvWrapper({'env_config': {'custom_options': {'num_players': players, 'history_len': slots}}})
    obs, _ = env.reset(seed=2)
    cls = NlHoldemLgNet if large else NlHoldemNet
    model = cls(env.observation_space, env.action_space, 5, {}, 'test')
    tensors = {k: torch.tensor(v).unsqueeze(0) for k,v in obs.items()}
    logits, _ = model({'obs': tensors}, [], None)
    assert logits.shape == (1,5)
    assert model.value_function().shape == (1,)
    assert torch.isfinite(logits).all()
    assert obs['legal_moves'][logits.argmax().item()] == 1


@pytest.mark.parametrize('second_total,reopens', [(150,False),(200,True)])
def test_short_allins_only_reopen_after_a_full_cumulative_raise(second_total, reopens):
    from rlcard.games.nolimitholdem import Round, Player, Dealer
    rng = np.random.RandomState(4)
    players = [Player(i, 1000, rng) for i in range(6)]
    for p in players:
        p.bet(100)
    players[1].remained_chips = 30
    players[2].remained_chips = second_total - 100
    round = Round(6, 100, Dealer(rng), rng)
    round.start_new_round(0, [100]*6)
    round.proceed_round(players, Action.CHECK_CALL)
    round.proceed_round(players, Action.ALL_IN)
    round.proceed_round(players, Action.ALL_IN)
    for _ in range(3):
        round.proceed_round(players, Action.CHECK_CALL)
    assert round.game_pointer == 0
    assert (Action.ALL_IN in round.get_nolimit_legal_actions(players)) == reopens


def test_short_call_records_only_chips_actually_paid():
    from rlcard.games.nolimitholdem import Round, Player, Dealer
    rng = np.random.RandomState(4)
    players = [Player(i, 100, rng) for i in range(6)]
    players[0].remained_chips = 10
    round = Round(6, 2, Dealer(rng), rng)
    round.start_new_round(0, [0,50,0,0,0,0])
    round.proceed_round(players, Action.CHECK_CALL)
    assert round.raised[0] == 10
    assert players[0].in_chips == 10


def test_seed_repeats_cards_button_and_hero_seat():
    from agi.nl_holdem_env import NlHoldemEnvWithOpponent
    conf = {'env_config': {'custom_options': {'num_players': 6}}}
    env = NlHoldemEnvWithOpponent(conf, opponent='random')
    first, _ = env.reset(seed=82)
    hero = env.our_pid
    second, _ = env.reset(seed=82)
    assert env.our_pid == hero
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_latest_history_and_folded_status_survive_truncation():
    from agi.nl_holdem_env import NlHoldemEnvWrapper
    env = NlHoldemEnvWrapper({'env_config': {'custom_options': {'num_players': 6, 'history_len': 2}}})
    env.reset(seed=3)
    env.history[0] = [[0,1,[0,1]], [1,0,[0,1]], [2,3,[0,1,3]]]
    env.env.game.players[1].status = PlayerStatus.FOLDED
    obs = env._get_observation(env.last_obs)
    assert obs['action_info'][1,0,0] == 1
    assert obs['action_info'][2,3,1] == 1
    assert obs['table_info'][1,1] == 0


def test_stable_probabilities_and_deterministic_checkpoint_roundtrip(tmp_path):
    import pickle
    import torch
    from agi.nl_holdem_env import NlHoldemEnvWrapper
    from agi.evaluation_tools import NNAgent, action_probabilities
    probs = action_probabilities([1e30, 100001, 100000, 0, 0], [0,1,1,0,0])
    assert probs[0] == 0
    assert 0 < probs[2] < probs[1] < 1
    assert action_probabilities([0,10,0,0,0], [0,1,1,0,0], 10)[2] > action_probabilities([0,10,0,0,0], [0,1,1,0,0])[2]
    conf = {'env_config': {'custom_options': {'num_players': 6}}, 'model': {'custom_model': 'NlHoldemNet'}}
    env = NlHoldemEnvWrapper(conf)
    obs, _ = env.reset(seed=2)
    agent = NNAgent(env.observation_space, env.action_space, conf, None)
    path = tmp_path / 'six.pkl'
    with path.open('wb') as f:
        pickle.dump({k:v.numpy() for k,v in agent.model.state_dict().items()}, f)
    loaded = NNAgent(env.observation_space, env.action_space, conf, path)
    assert len({loaded.make_action(obs, deterministic=True) for _ in range(10)}) == 1
    with pytest.raises(ValueError, match='Incompatible checkpoint'):
        NNAgent(env.observation_space, env.action_space, conf, 'weights/c_1048.pkl')


def test_six_player_random_hands_conserve_chips_and_never_act_folded_or_allin():
    rng = np.random.default_rng(9)
    for seed in range(300):
        env = rlcard.make('no-limit-holdem', config={'game_num_players': 6, 'seed': seed})
        env.reset()
        for p in env.game.players:
            p.remained_chips = int(rng.integers(10,201)) - p.in_chips
        total = sum(p.in_chips + p.remained_chips for p in env.game.players)
        actions = 0
        while not env.is_over():
            pid = env.get_player_id()
            assert env.game.players[pid].status == PlayerStatus.ALIVE
            legal = env.game.get_legal_actions()
            assert legal
            env.step(rng.choice(legal).value)
            assert sum(p.in_chips + p.remained_chips for p in env.game.players) == total
            assert all(p.remained_chips >= 0 for p in env.game.players)
            actions += 1
            assert actions < 200
        assert env.get_payoffs().sum() == 0
        if sum(p.status != PlayerStatus.FOLDED for p in env.game.players) > 1:
            assert len(env.game.public_cards) == 5


def test_heads_up_button_is_small_blind_and_acts_first_preflop():
    env = rlcard.make('no-limit-holdem', config={'game_num_players': 2, 'dealer_id': 0})
    env.reset()
    assert env.game.players[0].in_chips == 1
    assert env.get_player_id() == 0
    env.step(1)
    env.step(1)
    assert env.get_player_id() == 1


def test_fold_ends_training_episode_only_after_opponents_finish():
    from agi.nl_holdem_env import NlHoldemEnvWithOpponent
    env = NlHoldemEnvWithOpponent({'env_config': {'custom_options': {'num_players': 6}}}, opponent='random')
    env.reset(seed=23)
    _, reward, done, _, _ = env.step(0)
    assert done and env.env.is_over()
    assert reward == env.last_reward / env.reward_scale



def test_history_board_api_matches_observation_inference():
    from agi.nl_holdem_env import NlHoldemEnvWrapper
    from agi.evaluation_tools import NNAgent
    conf = {'env_config': {'custom_options': {'num_players': 6}}, 'model': {'custom_model': 'NlHoldemNet'}}
    env = NlHoldemEnvWrapper(conf)
    env.reset(seed=20)
    obs, _, _, _, _ = env.step(0)
    agent = NNAgent(env.observation_space, env.action_space, conf, None)
    raw = env.last_obs[0]['raw_obs']
    status = {PlayerStatus.ALIVE: 'alive', PlayerStatus.FOLDED: 'folded', PlayerStatus.ALLIN: 'allin'}
    board = dict(hand=raw['hand'], public_cards=raw['public_cards'], stakes=raw['stakes'],
                 contributions=[p.in_chips for p in env.env.game.players],
                 statuses=[status[p.status] for p in env.env.game.players],
                 current_player=env.my_agent(), dealer_id=env.env.game.dealer_id,
                 legal_moves=list(env.last_obs[0]['legal_actions']))
    assert agent.model_deci(env.history, board, deterministic=True) == agent.make_action(obs, deterministic=True)


def test_step_back_restores_street_and_round_dealer_reference():
    env = rlcard.make('no-limit-holdem', config={'game_num_players': 6, 'seed': 4, 'allow_step_back': True})
    env.reset()
    first = env.get_player_id()
    env.step(1)
    env.step_back()
    assert env.get_player_id() == first
    assert env.game.round.dealer is env.game.dealer
    assert env.game.stage.value == 0


def test_split_pot_odd_chip_goes_left_of_button():
    env = rlcard.make('no-limit-holdem', config={'game_num_players': 6, 'dealer_id': 0})
    env.reset()
    for i,p in enumerate(env.game.players):
        p.in_chips = 1 if i < 3 else 0
        p.status = PlayerStatus.FOLDED if i >= 2 else PlayerStatus.ALIVE
        p.hand = [Card('H','2'), Card('D','3')]
    env.game.public_cards = [Card('S',r) for r in 'TJQKA']
    assert env.get_payoffs().tolist() == [0,1,-1,0,0,0]


def test_gui_handles_folded_hero_and_resends_last_message():
    from gui import play_against_ai_in_ui as ui
    class CheckCaller:
        def make_action(self, obs, deterministic=False):
            return 1
    ui.nn_agent = CheckCaller()
    thread = ui.MyThread()
    client = ui.socketio.test_client(ui.app)
    thread.run()
    assert thread.env.my_agent() == 0
    thread.send_message({'action_id': 0})
    assert thread.env.env.game.is_over()
    thread.resend_last_message()
    assert client.get_received()
    client.disconnect()



def test_lone_matched_big_blind_is_not_given_a_fold_decision():
    env = rlcard.make('no-limit-holdem', config={'game_num_players': 6, 'dealer_id': 0})
    env.reset()
    env.game.players[1].remained_chips = 1
    for _ in range(4):
        env.step(0)
    env.step(1)
    assert env.is_over()
    assert len(env.game.public_cards) == 5
