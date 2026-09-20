import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray.rllib.algorithms.impala.impala_torch_policy import ImpalaTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
import copy
from io import StringIO 
import sys
torch, _ = try_import_torch()
import rlcard
from rlcard.utils import set_seed
import random

color2ind = dict(zip("CDHS",[0,1,2,3]))
rank2ind = dict(zip("23456789TJQKA",[0,1,2,3,4,5,6,7,8,9,10,11,12]))

class NlHoldemEnvWrapper(gym.Env):
    def __init__(self,policy_config,weights=None):
        super().__init__()
        self.policy_config = policy_config
        seed = random.randint(0,1000000)
        self.env = rlcard.make(
            'no-limit-holdem',
            config={
                'seed': seed,
                'game_num_players': 6,
                'allow_num_players': 6,
            }
        )
        set_seed(seed)
        self.action_num = 5
        
        
        space = {
                'card_info': spaces.Box(low=-1024, high=1024, shape=(4,13,6)),
                'action_info': spaces.Box(low=-256, high=256, shape=(8,self.action_num,4 * 20 + 1)),
                'extra_info': spaces.Box(low=-256, high=256, shape=(6,)),
                'legal_moves': spaces.Box(
                    low=-1,
                    high=1,
                    shape=list(
                        [self.action_num,]
                    )
                ),
            }
        
        self.observation_space = spaces.Dict(space)
        self.action_space = spaces.Discrete(self.action_num)

    def _get_observation(self,obs):
        card_info = np.zeros([4,13,6],np.uint8)
        action_info = np.zeros([8,self.action_num,4 * 20 + 1],np.uint8)
        extra_info = np.zeros([6],np.uint8)
        legal_actions_info = np.zeros([self.action_num],np.uint8) # 25 channel
        
        hold_card = obs[0]["raw_obs"]["hand"]
        public_card = obs[0]["raw_obs"]["public_cards"]
        current_legal_actions = [i.value for i in obs[0]["raw_obs"]["legal_actions"]]
        
        for ind in current_legal_actions:
            legal_actions_info[ind] = 1
        
        flop_card = public_card[:3]
        turn_card = public_card[3:4]
        river_card = public_card[4:5]
        
        for one_card in hold_card:
            card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][0] = 1
            
        for one_card in flop_card:
            card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][1] = 1
            
        for one_card in turn_card:
            card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][2] = 1
            
        for one_card in river_card:
            card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][3] = 1
            
        for one_card in public_card:
            card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][4] = 1
            
        for one_card in public_card + hold_card:
            card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][5] = 1
            
        
        for ind_round,one_history in enumerate(self.history):
            for ind_h,(player_id,action_id,legal_actions) in enumerate(one_history[:20]):
                action_info[player_id,action_id,ind_round * 20 + ind_h] = 1
                action_info[6,action_id,ind_round * 20 + ind_h] = 1
                
                for la_ind in legal_actions:
                    action_info[7,la_ind,ind_round * 20 + ind_h] = 1
                    
        action_info[:,:,-1] = self.my_agent()
        
        for i in range(6):
            extra_info[i] = obs[0]["raw_obs"]["stakes"][i]
        
        return {
            "card_info": card_info,
            "action_info": action_info,
            "legal_moves": legal_actions_info,
            "extra_info": extra_info,
        }
    
    def _log_action(self,action_ind):
        self.history[
            self.last_obs[0]["raw_obs"]["stage"].value
        ].append([
            self.last_obs[0]["raw_obs"]["current_player"],
            action_ind,
            [x.value for x in self.last_obs[0]["raw_obs"]["legal_actions"]]
        ])
    
    def my_agent(self):
        return self.env.get_player_id()
    
    def convert(self,reward):
        return float(reward)
        
    def _inner_step(self, action):
        """Internal step returning old-style 4-tuple for use by opponent logic."""
        self._log_action(action)
        obs = self.env.step(action)
        self.last_obs = obs
        obs = self._get_observation(obs)
        
        done = False
        reward = [0 for _ in range(6)]
        info = {}
        if self.env.game.is_over():
            done = True
            reward = list(self.env.get_payoffs())
            
        return obs,reward,done,info

    def step(self, action):
        obs, reward, done, info = self._inner_step(action)
        return obs, reward, done, False, info

    def _inner_reset(self):
        """Internal reset returning old-style single obs for use by opponent logic."""
        self.history = [[],[],[],[]]
        obs = self.env.reset()
        self.last_obs = obs
        return self._get_observation(obs)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        obs = self._inner_reset()
        return obs, {}
    
    def legal_moves(self):
        pass
    

class NlHoldemEnvWithOpponent(NlHoldemEnvWrapper):
    def __init__(self,policy_config,weights=None,opponent="nn"):
        super(NlHoldemEnvWithOpponent, self).__init__(policy_config,weights)
        self.opponent = opponent
        self.rwd_ratio = policy_config["env_config"]["custom_options"].get("rwd_ratio",1)
        self.is_done = False
        if self.opponent == "nn":
            self.oppo_name = None
            self.oppo_preprocessor = ModelCatalog.get_preprocessor_for_space(self.observation_space, policy_config.get("model"))
            
            from ray.rllib.algorithms.impala import ImpalaConfig
            dummy_config = ImpalaConfig().framework("torch")
            dummy_config.model = policy_config.get("model", {})
            dummy_config.env_config = policy_config.get("env_config", {})
            dummy_config = dummy_config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            dummy_config = dummy_config.to_dict() # old API compatibility for Policy constructor
            self.oppo_policy = ImpalaTorchPolicy(
                observation_space=self.oppo_preprocessor.observation_space,
                action_space=self.action_space,
                config=dummy_config,
            )
            if weights is not None:
                import pickle
                with open(weights,'rb') as fhdl:
                    weights = pickle.load(fhdl)
                self.oppo_policy.set_weights(weights)

    def _opponent_step(self,obs):
        if self.opponent == "random":
            rwd = [0 for _ in range(6)]
            done = False
            info = {}
            while self.my_agent() != self.our_pid:
                legal_moves = obs["legal_moves"]
                action_ind = np.random.choice(np.where(legal_moves)[0])
                obs,rwd,done,info = super(NlHoldemEnvWithOpponent, self)._inner_step(action_ind)
                if done:
                    break
            return obs,rwd,done,info
        elif self.opponent == "nn":
            rwd = [0 for _ in range(6)]
            done = False
            info = {}
            while self.my_agent() != self.our_pid:
                observation = self.oppo_preprocessor.transform(obs)
                action_ind = self.oppo_policy.compute_actions([observation])[0][0]
                obs,rwd,done,info = super(NlHoldemEnvWithOpponent, self)._inner_step(action_ind)
                if done:
                    break
            return obs,rwd,done,info
        else:
            raise        
        
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.last_reward = 0
        self.is_done = False
        self.our_pid = random.randint(0,5)
        
        obs = super(NlHoldemEnvWithOpponent, self)._inner_reset()
        
        while True:
            obs,rwd,done,info = self._opponent_step(obs)
            if not done:
                return obs, {}
            else:
                obs = super(NlHoldemEnvWithOpponent, self)._inner_reset()
            
    def step(self,action):
        obs,reward,done,info = super(NlHoldemEnvWithOpponent, self)._inner_step(action)
        reward = [i * self.rwd_ratio for i in reward]
        if done:
            self.is_done = True
            self.last_reward = reward[self.our_pid]
            return obs,reward[self.our_pid],done,False,info
        else:
            obs,reward,done,info = self._opponent_step(obs)
            reward = [i * self.rwd_ratio for i in reward]
            if done:
                self.is_done = True
                self.last_reward = reward[self.our_pid]
            return obs,reward[self.our_pid],done,False,info
        