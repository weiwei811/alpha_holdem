import gymnasium as gym
import numpy as np
from gymnasium import spaces
import rlcard
import random

color2ind = dict(zip("CDHS",[0,1,2,3]))
rank2ind = dict(zip("23456789TJQKA",[0,1,2,3,4,5,6,7,8,9,10,11,12]))

class NlHoldemEnvWrapper(gym.Env):
    def __init__(self,policy_config,weights=None):
        super().__init__()
        self.policy_config = policy_config
        seed = random.randint(0,1000000)
        self.num_players = policy_config.get("env_config", {}).get("custom_options", {}).get("num_players", 6)
        self.env = rlcard.make(
            'no-limit-holdem',
            config={
                'seed': seed,
                'game_num_players': self.num_players,
                'allow_num_players': self.num_players,
            }
        )
        if not 2 <= self.num_players <= 6:
            raise ValueError('num_players must be between 2 and 6')
        self.betting_features = policy_config.get('env_config', {}).get('custom_options', {}).get('betting_features', False)
        self.action_num = 5
        
        # History slots per round and action_info channels depend on num_players
        self.history_slots = policy_config.get("env_config", {}).get("custom_options", {}).get("history_len", 20 if self.num_players > 2 else 6)
        if self.history_slots < 1:
            raise ValueError('history_len must be positive')
        self.action_info_rows = self.num_players + 2  # player channels + aggregate + legal actions
        
        space = {
                'card_info': spaces.Box(low=0, high=1, shape=(4,13,6), dtype=np.float32),
                'action_info': spaces.Box(low=0, high=1, shape=(self.action_info_rows, self.action_num, 4 * self.history_slots + 1), dtype=np.float32),
                'extra_info': spaces.Box(low=0, high=np.inf, shape=(self.num_players,), dtype=np.float32),
                'table_info': spaces.Box(low=0, high=np.inf, shape=(self.num_players, 5), dtype=np.float32),
                'legal_moves': spaces.Box(
                    low=-1,
                    high=1,
                    shape=(self.action_num,), dtype=np.float32
                ),
            }
        
        if self.betting_features:
            space['betting_info'] = spaces.Box(low=0, high=np.inf, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Dict(space)
        self.action_space = spaces.Discrete(self.action_num)

    def _get_observation(self,obs):
        card_info = np.zeros([4,13,6],np.float32)
        action_info = np.zeros([self.action_info_rows, self.action_num, 4 * self.history_slots + 1],np.float32)
        extra_info = np.zeros([self.num_players],np.float32)
        legal_actions_info = np.zeros([self.action_num],np.float32)
        
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
            for ind_h,(player_id,action_id,legal_actions) in enumerate(one_history[-self.history_slots:]):
                action_info[player_id,action_id,ind_round * self.history_slots + ind_h] = 1
                action_info[self.num_players,action_id,ind_round * self.history_slots + ind_h] = 1
                
                for la_ind in legal_actions:
                    action_info[self.num_players + 1,la_ind,ind_round * self.history_slots + ind_h] = 1
                    
        action_info[self.my_agent(), :, -1] = 1
        
        for i in range(self.num_players):
            extra_info[i] = obs[0]["raw_obs"]["stakes"][i]
        
        from rlcard.games.limitholdem import PlayerStatus
        table_info = np.zeros((self.num_players, 5), dtype=np.float32)
        for i, player in enumerate(self.env.game.players):
            table_info[i] = [player.in_chips, player.status != PlayerStatus.FOLDED,
                             player.status == PlayerStatus.ALLIN,
                             i == self.env.game.dealer_id, i == self.my_agent()]
        result = {
            "table_info": table_info,
            "card_info": card_info,
            "action_info": action_info,
            "legal_moves": legal_actions_info,
            "extra_info": extra_info,
        }
        if self.betting_features:
            result['betting_info'] = self._betting_information(obs[0]['raw_obs'])
        return result

    def _betting_information(self, raw):
        # Only public state; use street commitments rather than total commitments
        # because short all-ins from earlier streets can have smaller totals.
        game = self.env.game
        actor = int(raw['current_player'])
        stacks = np.asarray(raw['stakes'], dtype=np.float32)
        pot = float(sum(p.in_chips for p in game.players))
        call = max(0.0, float(max(game.round.raised) - game.round.raised[actor]))
        legal = [a.value for a in raw['legal_actions']]
        if not legal:
            call = 0.0
        cost = min(call, float(stacks[actor]))
        from rlcard.games.limitholdem import PlayerStatus
        opponents = [i for i,p in enumerate(game.players)
                     if i != actor and p.status != PlayerStatus.FOLDED]
        effective = min(float(stacks[actor]), max((float(stacks[i]) for i in opponents), default=0.0))
        public_cards = len(raw['public_cards'])
        street = 0 if public_cards == 0 else public_cards - 2
        return np.asarray([pot, call, cost, game.round.last_raise_amount,
                           stacks[actor], effective, cost / max(pot + cost, 1.0),
                           effective / max(pot, 1.0), len(opponents),
                           street, any(a >= 2 for a in legal),
                           any(game.players[i].status == PlayerStatus.ALLIN for i in opponents)], dtype=np.float32)

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
        if self.env.game.is_over():
            raise RuntimeError('Reset before stepping a completed hand')
        if int(action) not in self.last_obs[0]['legal_actions']:
            raise ValueError('Illegal action: {}'.format(action))
        self._log_action(int(action))
        obs = self.env.step(action)
        self.last_obs = obs
        obs = self._get_observation(obs)
        
        done = False
        reward = [0 for _ in range(self.num_players)]
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
        super().reset(seed=seed)
        if seed is not None:
            self.env.seed(seed)
        obs = self._inner_reset()
        return obs, {}
    
    def legal_moves(self):
        pass
    

class NlHoldemEnvWithOpponent(NlHoldemEnvWrapper):
    def __init__(self,policy_config,weights=None,opponent="nn"):
        super(NlHoldemEnvWithOpponent, self).__init__(policy_config,weights)
        self.opponent = opponent
        self.rwd_ratio = policy_config["env_config"]["custom_options"].get("rwd_ratio",1)
        self.reward_scale = float(policy_config['env_config']['custom_options'].get('reward_scale', 200.0))
        if self.reward_scale <= 0 or not np.isfinite(self.reward_scale):
            raise ValueError('reward_scale must be finite and positive')
        self.is_done = False
        self.prepare_opponent = None
        self.on_hand_end = None
        if self.opponent == "nn":
            from ray.rllib.algorithms.impala.impala_torch_policy import ImpalaTorchPolicy
            from ray.rllib.models import ModelCatalog
            self.oppo_name = None
            self.oppo_preprocessor = ModelCatalog.get_preprocessor_for_space(self.observation_space, policy_config.get("model"))
            
            from ray.rllib.algorithms.impala import ImpalaConfig
            dummy_config = ImpalaConfig().framework("torch").resources(num_gpus=0)
            dummy_config.model.update(policy_config.get("model", {}))
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
            rwd = [0 for _ in range(self.num_players)]
            done = False
            info = {}
            while self.my_agent() != self.our_pid:
                legal_moves = obs["legal_moves"]
                action_ind = self.np_random.choice(np.where(legal_moves)[0])
                obs,rwd,done,info = super(NlHoldemEnvWithOpponent, self)._inner_step(action_ind)
                if done:
                    break
            return obs,rwd,done,info
        elif self.opponent == "nn":
            rwd = [0 for _ in range(self.num_players)]
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
            raise ValueError('Unknown opponent: {}'.format(self.opponent))
        
    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.env.seed(seed)
        if self.prepare_opponent is not None:
            self.prepare_opponent(self)
        self.last_reward = 0
        self.is_done = False
        self.our_pid = int(self.np_random.integers(self.num_players))
        
        obs = super(NlHoldemEnvWithOpponent, self)._inner_reset()
        
        while True:
            obs,rwd,done,info = self._opponent_step(obs)
            if not done:
                return obs, {}
            else:
                self._record_hand(rwd)
                # No learner action exists, but the dealt hand still counts in league profit.
                self.our_pid = int(self.np_random.integers(self.num_players))
                obs = super(NlHoldemEnvWithOpponent, self)._inner_reset()
            
    def _record_hand(self, reward):
        self.last_reward = float(reward[self.our_pid])
        if self.on_hand_end is not None:
            self.on_hand_end(self, self.last_reward)

    def step(self, action):
        obs, reward, done, info = super()._inner_step(action)
        if not done:
            obs, reward, done, info = self._opponent_step(obs)
        if done:
            self.is_done = True
            self._record_hand(reward)
        learner_reward = float(reward[self.our_pid]) * self.rwd_ratio / self.reward_scale
        return obs, learner_reward, done, False, info
