import ray
import logging
import argparse
import ast
from utils import get_winrate_and_weight, save_learner_weights
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
import pickle
import os

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env

from agi.nl_holdem_env import NlHoldemEnvWithOpponent
from agi.nl_holdem_net import NlHoldemNet
from agi.nl_holdem_lg_net import NlHoldemLgNet

ModelCatalog.register_custom_model('NlHoldemNet', NlHoldemNet)
ModelCatalog.register_custom_model('NlHoldemLgNet', NlHoldemLgNet)

from agi.league import League

def main():
    parser = argparse.ArgumentParser(description="Train AlphaNLHoldem via League Training.")
    parser.add_argument('--conf',  type=str, default='confs/nl_holdem.py', help="Path to the training config file (e.g., confs/nl_holdem.py).")
    parser.add_argument('--gap',  type=int, default=1000, help="Number of iterations between checking if a new historical agent should be saved.")
    parser.add_argument('--sp',  type=float, default=0.0, help="Probability of playing against the agent's own current policy (self-play).")
    parser.add_argument('--exg_oppo_prob',  type=float, default=0.01, help="Probability of exchanging/sampling a new opponent after an episode.")
    parser.add_argument('--upwin',  type=float, default=1.0, help="Win rate threshold required against historical agents to save a new checkpoint.")
    parser.add_argument('--kbest',  type=int, default=5, help="Number of best historical agents to consider in the league evaluations.")
    parser.add_argument('--league_tracker_n',  type=int, default=10000, help="Number of games to track in the league statistics.")
    parser.add_argument('--last_num',  type=int, default=100000, help="The number of latest matches to consider when evaluating win rates.")
    parser.add_argument('--rwd_update_ratio',  type=float, default=1.0, help="Ratio for updating results (rewards) to the league tracker.")
    parser.add_argument('--restore',  type=str, default=None, help="Directory to restore training from (e.g., league/history_agents).")
    parser.add_argument('--output_dir',  type=str, default="league/history_agents", help="Directory where historical agents will be saved.")
    parser.add_argument('--mode', type=str, default="local", help="Ray cluster mode to use, usually 'local'.")
    parser.add_argument('--experiment_name', default='run_trial_1', type=str, help="Name of the experiment run.")
    parser.add_argument('--iterations', type=int, default=0, help='Stop after N iterations; 0 trains continuously.')
    parser.add_argument('--workers', type=int, default=None, help='Override number of rollout workers.')
    parser.add_argument('--gpus', type=int, default=None, help='Legacy CUDA count: 0 or 1.')
    parser.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()

    if args.gap < 1 or args.iterations < 0 or args.league_tracker_n < 1 or args.kbest < 1:
        parser.error('gap, league_tracker_n and kbest must be positive; iterations must be nonnegative')
    for probability in (args.sp, args.exg_oppo_prob, args.rwd_update_ratio):
        if not 0 <= probability <= 1:
            parser.error('Probabilities must be between 0 and 1')
    conf = ast.literal_eval(open(args.conf).read())
    if args.workers is not None:
        conf['num_env_runners'] = args.workers
    if args.batch_size is not None:
        conf['train_batch_size'] = args.batch_size
        conf['minibatch_size'] = min(args.batch_size, conf.get('minibatch_size', args.batch_size))

    from agi.devices import configure_training_device
    try:
        device = configure_training_device(conf, args.device, args.gpus)
    except ValueError as error:
        parser.error(str(error))
    print('Learner device: {}'.format(device), flush=True)
    if device.type == 'mps':
        print('MPS training is experimental; use --device cpu if an operation is unsupported.', flush=True)
    if args.mode == 'local':
        ray.init(include_dashboard=False)
    else:
        parser.error('Only local mode is supported')

    register_env("NlHoldemEnvWithOpponent", lambda config: NlHoldemEnvWithOpponent(
            conf
    ))

    league = League.remote(
        n=args.league_tracker_n,
        last_num=args.last_num,
        kbest=args.kbest,
        output_dir=args.output_dir,
    )

    class LeagueCallbacks(DefaultCallbacks):
        def __init__(self):
            super().__init__()
            self.count = 0

        def on_sub_environment_created(self, *, worker, sub_environment, **kwargs):
            def prepare_opponent(env):
                if env.oppo_name is not None:
                    return
                # Invoked by reset after worker policies exist, before any actions.
                if not ray.get(league.initized.remote()):
                    weight = worker.get_policy("default_policy").get_weights()
                    ray.get(league.initize_if_possible.remote(weight))
                pid, weight = ray.get(league.select_opponent.remote())
                env.oppo_policy.set_weights(weight)
                env.oppo_name = pid

            def record_hand(env, reward):
                if np.random.random() < args.rwd_update_ratio:
                    ray.get(league.update_result.remote(
                        None if env.oppo_name == "self" else env.oppo_name,
                        reward, selfplay=env.oppo_name == "self"))

            sub_environment.prepare_opponent = prepare_opponent
            sub_environment.on_hand_end = record_hand

        def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
            pass

        def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
            default_policy = policies["default_policy"]
            for env in [base_env.get_sub_environments()[env_index]]:
                if hasattr(env, "is_done") and env.is_done:
                    # 2. 更新对手权重
                    if np.random.random() < args.exg_oppo_prob:
                        if np.random.random() < args.sp:
                            p_weights = default_policy.get_weights()
                            weight = {}
                            for k,v in p_weights.items():
                                k = k.replace("default_policy","oppo_policy")
                                weight[k] = v
                            env.oppo_name = "self"
                            if hasattr(env, "oppo_policy"):
                                env.oppo_policy.set_weights(weight)
                        else:
                            pid, weight = ray.get(league.select_opponent.remote())
                            env.oppo_name = pid
                            if hasattr(env, "oppo_policy"):
                                env.oppo_policy.set_weights(weight)

        def on_train_result(self, *, algorithm, result, **kwargs):
            save_learner_weights(args.output_dir, algorithm.get_policy().get_weights())
            winrates_pd = ray.get(league.get_statics_table.remote())
            winrates_pd.to_csv("winrates.csv", header=False, index=False)

            table_t = winrates_pd.T
            table_t["mbb/h"] = np.asarray(table_t["winrate"] / 2.0 * 1000.0, dtype=int)

            # Print the table fully to the terminal to bypass Ray Tune's truncation
            print("\n" + "="*50)
            print("CURRENT LEAGUE WINRATES:")
            print(table_t.T.to_string())
            print("="*50 + "\n")

            result['league_mean_chip_profit'] = {str(k): float(v) for k,v in zip(table_t['oppo'], table_t['winrate'])}
            self.count += 1

            gap = args.gap
            if ray.get(league.winrate_all_match.remote(args.upwin)) or self.count % gap == 0:
                p_weights = algorithm.get_policy("default_policy").get_weights()
                weight = {}
                for k,v in p_weights.items():
                    k = k.replace("default_policy","oppo_policy")
                    weight[k] = v
                ray.get(league.add_weight.remote(weight))
                if not os.path.exists("weights"):
                    os.makedirs("weights")
                with open('output_weight.pkl','wb') as whdl:
                    pickle.dump(weight,whdl)
                with open('weights/output_weight_{}.pkl'.format(self.count),'wb') as whdl:
                    pickle.dump(weight,whdl)


    from ray.rllib.algorithms.impala import ImpalaConfig
    from agi.portable_impala import PortableImpala
    config = (ImpalaConfig(algo_class=PortableImpala).update_from_dict(conf)
              .api_stack(enable_rl_module_and_learner=False,
                         enable_env_runner_and_connector_v2=False)
              .callbacks(LeagueCallbacks))
    restore_weights = None
    if args.restore is not None:
        restore_weights = get_winrate_and_weight(args.restore, league)
    agent = None
    try:
        logdir = os.path.abspath(os.path.join('work', 'ray_results', args.experiment_name))
        os.makedirs(logdir, exist_ok=True)
        agent = config.build_algo(logger_creator=lambda config: UnifiedLogger(config, logdir, loggers=None))
        actual_device = next(agent.get_policy().model.parameters()).device
        if actual_device.type != device.type:
            raise RuntimeError('Requested {} but learner is on {}'.format(device, actual_device))
        print('Verified learner model on {}'.format(actual_device), flush=True)
        if restore_weights is not None:
            agent.get_policy().set_weights(restore_weights)
            agent.env_runner_group.sync_weights()
            print("Restored learner weights from " + args.restore, flush=True)
        iteration = 0
        while args.iterations == 0 or iteration < args.iterations:
            result = agent.train()
            iteration += 1
            print('Training iteration {} completed'.format(iteration), flush=True)
        os.makedirs(args.output_dir, exist_ok=True)
        save_learner_weights(args.output_dir, agent.get_policy().get_weights())
        with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as stream:
            import json
            json.dump(conf, stream, indent=2)
        ray.get(league.get_statics_table.remote())
        print('Checkpoint saved to ' + os.path.join(args.output_dir, 'output_weight.pkl'))
    finally:
        if agent is not None:
            agent.stop()
        ray.shutdown()


if __name__ == '__main__':
    main()
