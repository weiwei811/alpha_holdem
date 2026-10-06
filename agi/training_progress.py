"""Progress-based scheduling and reproducible held-out evaluation."""
import numpy as np


class StepInterval:
    def __init__(self, interval, previous=0):
        self.interval = interval
        self.previous = previous

    def due(self, steps):
        if self.interval and steps >= self.previous + self.interval:
            self.previous = steps
            return True
        return False


def validate_batches(conf):
    fragment = conf.get('rollout_fragment_length', 50)
    for key in ('train_batch_size', 'minibatch_size'):
        size = conf.get(key)
        if size is not None and (size < fragment or size % fragment):
            raise ValueError(f'{key} must be positive and divisible by rollout_fragment_length ({fragment})')


def evaluate_weights(conf, weights, baseline, hands=600, seed=700000):
    from agi.evaluation_tools import NNAgent
    from agi.nl_holdem_env import NlHoldemEnvWrapper
    env = NlHoldemEnvWrapper(conf)
    hero_agent = NNAgent(env.observation_space, env.action_space, conf, weights, device='cpu')
    initial_agent = NNAgent(env.observation_space, env.action_space, conf, baseline, device='cpu')
    results = {}
    for opponent in ('random', 'check_call', 'initial'):
        rewards = []
        for hand in range(hands):
            hero = hand % env.num_players
            obs, _ = env.reset(seed=seed + hand)
            rng = np.random.default_rng(seed + hands + hand)
            hero_agent.rng = np.random.default_rng(seed + 2 * hands + hand)
            initial_agent.rng = np.random.default_rng(seed + 3 * hands + hand)
            done = False
            while not done:
                if env.my_agent() == hero:
                    action = hero_agent.make_action(obs)
                elif opponent == 'initial':
                    action = initial_agent.make_action(obs)
                elif opponent == 'check_call' and obs['legal_moves'][1]:
                    action = 1
                else:
                    action = int(rng.choice(np.flatnonzero(obs['legal_moves'])))
                obs, payoff, done, _, _ = env.step(action)
            rewards.append(float(payoff[hero]))
        mean = float(np.mean(rewards))
        radius = 1.96 * float(np.std(rewards, ddof=1)) / np.sqrt(hands)
        results[opponent] = {'hands': hands, 'chips_per_hand': mean,
                             'approx_95pct_ci': [mean - radius, mean + radius]}
    return results


def validate_restore_configuration(conf, directory):
    """Reject semantic/model changes before Ray loads an incompatible league."""
    import json
    from pathlib import Path
    path = Path(directory) / 'training_config.json'
    if not path.exists():
        if conf.get('model', {}).get('custom_model') == 'NlHoldemStructuredNet':
            raise ValueError('Structured model restore requires its matching training_config.json; train a fresh model')
        return
    original = json.loads(path.read_text())
    def signature(config):
        model = config.get('model', {})
        options = config.get('env_config', {}).get('custom_options', {})
        players = options.get('num_players', 6)
        return (model.get('custom_model', 'NlHoldemNet'), model.get('custom_model_config', {}),
                players, options.get('history_len', 20 if players > 2 else 6),
                options.get('betting_features', False), options.get('reward_scale', 200.0))
    if signature(conf) != signature(original):
        raise ValueError('Restore model/observation/reward config differs from checkpoint; use its matching config or train fresh')
