"""Evaluate a newly trained six-player policy against random legal actions."""
import argparse
import ast
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agi.nl_holdem_env import NlHoldemEnvWrapper
from agi.evaluation_tools import NNAgent, death_match


class RandomAgent:
    def make_action(self, obs, deterministic=False):
        return int(np.random.choice(np.flatnonzero(obs['legal_moves'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default=str(Path(__file__).resolve().parents[1] / 'confs/nl_holdem.py'))
    parser.add_argument('--weights', required=True)
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='cpu')
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    conf = ast.literal_eval(Path(args.conf).read_text())
    env = NlHoldemEnvWrapper(conf)
    agent = NNAgent(env.observation_space, env.action_space, conf, args.weights, device=args.device)
    rewards = death_match(agent, RandomAgent(), env, args.games, args.deterministic)
    print('Mean hero profit: {:.3f} chips/hand ({:.1f} mbb/hand)'.format(np.mean(rewards), np.mean(rewards)*500))


if __name__ == '__main__':
    main()
