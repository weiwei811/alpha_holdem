import ray
import gymnasium as gym
import logging
import argparse
from matplotlib import pyplot as plt
from utils import ProgressBar, ma_sample, get_winrate_and_weight, register_restore_weight_trainer
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
import pickle
import os
import pandas as pd

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray import tune

from agi.nl_holdem_env import NlHoldemEnvWithOpponent
from agi.nl_holdem_net import NlHoldemNet
from agi.nl_holdem_lg_net import NlHoldemLgNet

ModelCatalog.register_custom_model('NlHoldemNet', NlHoldemNet)
ModelCatalog.register_custom_model('NlHoldemLgNet', NlHoldemLgNet)

from agi.league import League
from ray.rllib.algorithms.impala.impala import Impala

parser = argparse.ArgumentParser(description="Train AlphaNLHoldem via League Training.")
parser.add_argument('--conf',  type=str, help="Path to the training config file (e.g., confs/nl_holdem.py).")
parser.add_argument('--gap',  type=int, default=1000, help="Number of iterations between checking if a new historical agent should be saved.")
parser.add_argument('--sp',  type=float, default=0.0, help="Probability of playing against the agent's own current policy (self-play).")
parser.add_argument('--exg_oppo_prob',  type=float, default=0.01, help="Probability of exchanging/sampling a new opponent after an episode.")
parser.add_argument('--upwin',  type=float, default=1.0, help="Win rate threshold required against historical agents to save a new checkpoint.")
parser.add_argument('--kbest',  type=int, default=5, help="Number of best historical agents to consider in the league evaluations.")
parser.add_argument('--league_tracker_n',  type=float, default=10000, help="Number of games to track in the league statistics.")
parser.add_argument('--last_num',  type=int, default=100000, help="The number of latest matches to consider when evaluating win rates.")
parser.add_argument('--rwd_update_ratio',  type=float, default=1.0, help="Ratio for updating results (rewards) to the league tracker.")
parser.add_argument('--restore',  type=str, default=None, help="Directory to restore training from (e.g., league/history_agents).")
parser.add_argument('--output_dir',  type=str, default="league/history_agents", help="Directory where historical agents will be saved.")
parser.add_argument('--mode', type=str, default="local", help="Ray cluster mode to use, usually 'local'.")
parser.add_argument('--experiment_name', default='run_trial_1', type=str, help="Name of the experiment run.")
args = parser.parse_args()

if args.mode == "local":
    ray.init()
else:
    raise RuntimeError("unknown mode: {}".format(args.mode))

conf = eval(open(args.conf).read().strip())

register_env("NlHoldemEnvWithOpponent", lambda config: NlHoldemEnvWithOpponent(
        conf
))

league = League.remote(
    n=args.league_tracker_n,
    last_num=args.last_num,
    kbest=args.kbest,
    output_dir=args.output_dir,
)

def get_train(weight):
    if weight is None:
        pweight = None
    else:
        pweight = {}
        for k,v in weight.items():
            k = k.replace("oppo_policy","default_policy")
            pweight[k] = v
            
    def train_fn_load(config):
        from ray.rllib.algorithms.impala import ImpalaConfig
        from ray import tune
        if isinstance(config, dict):
            config = ImpalaConfig().update_from_dict(config)
        
        config = config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        
        # Build the algorithm from the tuned config
        agent = config.build_algo()
        print("LOAD: after init, before load")

        if pweight is not None:
            agent.workers.local_worker().get_policy("default_policy").set_weights(pweight)
            agent.workers.sync_weights()

        print("LOAD: before train, after load")
        while True:
            result = agent.train()
            tune.report(result)
        agent.stop()

    return train_fn_load

if args.restore is not None:
    get_winrate_and_weight(args.restore,league)
    pid = ray.get(league.get_latest_policy_id.remote())
    print("latest pid: {}".format(pid))
    weight = ray.get(league.get_weight.remote(pid))
    train_func = get_train(weight)
else:
    train_func = get_train(None)


class LeagueCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.count = 0

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        default_policy = policies["default_policy"]
        # Eğer league ilk weight setine sahip değilse
        if not ray.get(league.initized.remote()):
            p_weights = default_policy.get_weights()
            weight = {}
            for k,v in p_weights.items():
                k = k.replace("default_policy","oppo_policy")
                weight[k] = v
            ray.get(league.initize_if_possible.remote(weight))
            
        for env in base_env.get_sub_environments():
            if getattr(env, "oppo_name", None) is None:
                pid, weight = ray.get(league.select_opponent.remote())
                env.oppo_name = pid
                if hasattr(env, "oppo_policy"):
                    env.oppo_policy.set_weights(weight)

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        pass

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        default_policy = policies["default_policy"]
        for env in base_env.get_sub_environments():
            if hasattr(env, "is_done") and env.is_done:
                # 1. 更新结果到league
                last_reward = env.last_reward
                pid = env.oppo_name
                
                if np.random.random() < args.rwd_update_ratio:
                    if pid == "self":
                        ray.get(league.update_result.remote(None, last_reward, selfplay=True))
                    else:
                        ray.get(league.update_result.remote(pid, last_reward, selfplay=False))

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
        winrates_pd = ray.get(league.get_statics_table.remote())
        winrates_pd.to_csv("winrates.csv", header=False, index=False)
        
        table_t = winrates_pd.T
        table_t["mbb/h"] = np.asarray(table_t["winrate"] / 2.0 * 1000.0, dtype=int)
        result['winrates'] = table_t.T
        self.count += 1
        
        gap = args.gap
        if ray.get(league.winrate_all_match.remote(args.upwin)) or self.count % gap == gap - 1:
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


tune_config = {
    "framework": "torch",
    'max_sample_requests_in_flight_per_worker': 1,
    "num_data_loader_buffers": 4,
    "callbacks": LeagueCallbacks,
    "num_gpus": 0,
    "_enable_rl_module_api": False,
    "_enable_learner_api": False,
}

tune_config.update(conf)

from ray.tune.execution.placement_groups import PlacementGroupFactory
bundles = [{"CPU": 1, "GPU": conf.get("num_gpus", 0)}]
for _ in range(conf.get("num_workers", 0)):
    bundles.append({"CPU": 1})
trainable = tune.with_resources(train_func, PlacementGroupFactory(bundles))

tune.run(
    trainable,
    config=tune_config,
    storage_path=os.path.abspath('log/'),
)
