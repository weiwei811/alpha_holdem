from ray.rllib.utils import try_import_tf
from ray.rllib.algorithms.impala.impala_tf_policy import ImpalaTF1Policy as VTraceTFPolicy
import pandas as pd
from ray.rllib.models import ModelCatalog
tf1, tf, tfv = try_import_tf()
tf = tf1 if tf1 else tf
tf.compat.v1.disable_eager_execution()
from tqdm import tqdm
import numpy as np

class NNAgent():
    def __init__(self,observation_space,action_space,policy_config,weights,variable_scope="oppo_policy"):
        # Merge user config with Impala defaults so newer required keys are present
        from ray.rllib.algorithms.impala import ImpalaConfig
        full_config = ImpalaConfig().to_dict()
        full_config.update(policy_config)
        # Ensure proper exploration and framework settings for TF1 graph mode
        full_config.setdefault("exploration_config", {"type": "StochasticSampling"})
        if not full_config.get("exploration_config", {}).get("type"):
            full_config["exploration_config"] = {"type": "StochasticSampling"}
        full_config.setdefault("framework", "tf")
        
        self.oppo_preprocessor = ModelCatalog.get_preprocessor_for_space(observation_space, full_config.get("model"))
        self.graph = tf.Graph()
        self.name = variable_scope
        with self.graph.as_default():
            with tf.variable_scope(variable_scope):
                self.oppo_policy = VTraceTFPolicy(
                    observation_space=self.oppo_preprocessor.observation_space,
                    action_space=action_space,
                    config=full_config,
                )
        if weights is not None:
            import pickle
            try:
                with open(weights,'rb') as fhdl:
                    weights_data = pickle.load(fhdl)
                new_weights = {}
                current_weights = self.oppo_policy.get_weights()
                for k, v in weights_data.items():
                    target_k = k.replace("oppo_policy", variable_scope)
                    if target_k in current_weights:
                        cur_shape = current_weights[target_k].shape
                        if v.shape == cur_shape:
                            new_weights[target_k] = v
                        elif len(v.shape) == len(cur_shape) and all(v.shape[i] <= cur_shape[i] for i in range(len(v.shape))):
                            # Pad smaller layer weights (e.g. from 2-player checkpoint)
                            padded = np.zeros(cur_shape, dtype=v.dtype)
                            slices = tuple(slice(0, s) for s in v.shape)
                            padded[slices] = v
                            new_weights[target_k] = padded
                self.oppo_policy.set_weights(new_weights)
            except Exception as e:
                print(f"Notice: Weight loading exception ({e}), proceeding with initialized weights.")
            
    def make_action(self,obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        observation = self.oppo_preprocessor.transform(obs)
        action_ind = self.oppo_policy.compute_actions(np.array([observation]))[0][0]
        return action_ind
    
def death_match(agent1,agent2,env):
    rewards = []
    for i in tqdm(range(5000)):
        obs = env.reset()
        d = False
        while not d:
            legal_moves = obs["legal_moves"]
            #action_ind = np.random.choice(np.where(legal_moves)[0])
            if env.my_agent() == 0:
                action_ind = agent1.make_action(obs)
            elif env.my_agent() == 1:
                action_ind = agent2.make_action(obs)
            else:
                raise
            obs,r,d,i = env.step(action_ind)
        rewards.append(r[0])
    
    for i in tqdm(range(5000)):
        obs = env.reset()
        d = False
        while not d:
            legal_moves = obs["legal_moves"]
            #action_ind = np.random.choice(np.where(legal_moves)[0])
            if env.my_agent() == 0:
                action_ind = agent2.make_action(obs)
            elif env.my_agent() == 1:
                action_ind = agent1.make_action(obs)
            else:
                raise
            obs,r,d,i = env.step(action_ind)
        rewards.append(r[1])
    return rewards