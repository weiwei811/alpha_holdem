"""PyTorch inference using the exact model and observations used in training."""
import pickle
from collections.abc import Mapping
import numpy as np
import torch
from agi.nl_holdem_net import NlHoldemNet
from agi.nl_holdem_lg_net import NlHoldemLgNet


def action_probabilities(logits, legal_moves, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    legal = np.asarray(legal_moves, dtype=bool)
    if not legal.any():
        raise ValueError('No legal moves in this observation')
    scores = np.asarray(logits, dtype=np.float64)
    if scores.shape != legal.shape or not np.isfinite(scores[legal]).all():
        raise ValueError('Invalid policy logits')
    shifted = (scores[legal] - scores[legal].max()) / temperature
    probabilities = np.zeros(scores.shape, dtype=np.float64)
    probabilities[legal] = np.exp(shifted)
    probabilities /= probabilities.sum()
    return probabilities


class NNAgent:
    def __init__(self, observation_space, action_space, policy_config, weights,
                 variable_scope='oppo_policy', seed=None, device='cpu'):
        from agi.devices import resolve_device
        self.device = resolve_device(device)
        self.policy_config = policy_config
        self.rng = np.random.default_rng(seed)
        config = policy_config.get('model', {})
        model_name = config.get('custom_model', 'NlHoldemNet')
        models = {'NlHoldemNet': NlHoldemNet, 'NlHoldemLgNet': NlHoldemLgNet}
        if model_name not in models:
            raise ValueError('Unknown inference model: ' + model_name)
        self.model = models[model_name](observation_space, action_space,
                                       action_space.n, config, variable_scope)
        if weights is not None:
            with open(weights, 'rb') as stream:
                loaded = pickle.load(stream)
            expected = self.model.state_dict()
            if (not isinstance(loaded, Mapping) or set(loaded) != set(expected) or
                    any(tuple(np.shape(loaded[k])) != tuple(expected[k].shape) for k in expected)):
                raise ValueError('Incompatible checkpoint. Heads-up TensorFlow weights cannot be '
                                 'padded into a six-player PyTorch policy; train a new six-player model.')
            self.model.load_state_dict({k: torch.as_tensor(v) for k,v in loaded.items()}, strict=True)
        self.model.to(self.device).eval()

    def make_action(self, obs, deterministic=False, temperature=1.0):
        if isinstance(obs, tuple):
            obs = obs[0]
        tensors = {k: torch.as_tensor(v, device=self.device).unsqueeze(0) for k,v in obs.items()}
        with torch.no_grad():
            logits, _ = self.model({'obs': tensors}, [], None)
        probs = action_probabilities(logits[0].cpu().numpy(), obs['legal_moves'], temperature)
        return int(probs.argmax() if deterministic else self.rng.choice(len(probs), p=probs))

    def model_deci(self, history, board, deterministic=False, temperature=1.0):
        """history: four street lists of (seat, action, legal_action_ids).

        board is a full public-state mapping: hand, public_cards, stakes,
        contributions, statuses ('alive', 'folded', 'allin'), current_player,
        dealer_id and legal_moves (action IDs). No opponents are merged.
        """
        from agi.nl_holdem_env import NlHoldemEnvWrapper
        from rlcard.games.limitholdem import PlayerStatus
        from rlcard.games.nolimitholdem import Action
        if len(history) != 4:
            raise ValueError('history must contain four street lists')
        env = NlHoldemEnvWrapper(self.policy_config)
        env.reset()
        n = env.num_players
        if any(len(board[k]) != n for k in ('stakes', 'contributions', 'statuses')):
            raise ValueError('Public seat state must include every player')
        actor = int(board['current_player'])
        dealer = int(board['dealer_id'])
        if not 0 <= actor < n or not 0 <= dealer < n:
            raise ValueError('Invalid current_player or dealer_id')
        statuses = {'alive': PlayerStatus.ALIVE, 'folded': PlayerStatus.FOLDED,
                    'allin': PlayerStatus.ALLIN}
        for i,p in enumerate(env.env.game.players):
            p.in_chips = board['contributions'][i]
            p.status = statuses[board['statuses'][i]]
        env.env.game.game_pointer = actor
        env.env.game.dealer_id = dealer
        env.history = history
        raw = dict(board, legal_actions=[Action(i) for i in board['legal_moves']])
        return self.make_action(env._get_observation(({'raw_obs': raw}, actor)),
                                deterministic=deterministic, temperature=temperature)


def model_deci(history, board, deterministic=False, *, agent, temperature=1.0):
    return agent.model_deci(history, board, deterministic, temperature)


def death_match(agent1, agent2, env, games=1000, deterministic=False):
    """Evaluate hero against copies of the other policy, rotating all seats."""
    rewards = []
    for game in range(games):
        hero = game % env.num_players
        obs, _ = env.reset()
        done = False
        while not done:
            agent = agent1 if env.my_agent() == hero else agent2
            action = agent.make_action(obs, deterministic=deterministic)
            obs, payoff, done, _, _ = env.step(action)
        rewards.append(payoff[hero])
    return rewards
