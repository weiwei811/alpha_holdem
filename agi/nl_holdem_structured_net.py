"""Actor-relative public features and independently encoded policy/value branches."""
import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from agi.nl_holdem_net import ResBlock


def actor_relative_observation(obs):
    table = obs['table_info'].float()
    players = table.shape[1]
    actor = table[:, :, 4].argmax(1)
    order = (actor[:, None] + torch.arange(players, device=actor.device)[None, :]) % players
    canonical = dict(obs)
    canonical['table_info'] = table.gather(1, order[:, :, None].expand(-1, -1, table.shape[2]))
    canonical['extra_info'] = obs['extra_info'].gather(1, order)
    actions = obs['action_info']
    player_rows = actions[:, :players].gather(1, order[:, :, None, None].expand(-1, -1, actions.shape[2], actions.shape[3]))
    canonical['action_info'] = torch.cat([player_rows, actions[:, players:]], dim=1)
    return canonical


class HoldemFeatureEncoder(nn.Module):
    def __init__(self, original, stack_scale, width=16):
        super().__init__()
        self.stack_scale = stack_scale
        self.players = original['extra_info'].shape[0]
        self.betting = 'betting_info' in original.spaces
        shape = original['action_info'].shape
        def convolution(channels):
            return nn.Sequential(ResBlock(channels, width, 3, 1),
                                 ResBlock(width, width*2, 3, 2),
                                 ResBlock(width*2, width*4, 3, 2))
        self.cards = convolution(6)
        self.actions = convolution(shape[-1])
        self.public = SlimFC(self.players * 6 + (12 if self.betting else 0), 32,
                             initializer=normc_initializer(1.0), activation_fn='relu')
        size = width*4*(4 + ((shape[0]+3)//4)*((shape[1]+3)//4)) + 32
        self.fuse = nn.Sequential(SlimFC(size, 256, initializer=normc_initializer(1.0), activation_fn='relu'),
                                  SlimFC(256, 128, initializer=normc_initializer(1.0), activation_fn='relu'),
                                  SlimFC(128, 64, initializer=normc_initializer(1.0), activation_fn='relu'))

    def forward(self, obs):
        table = obs['table_info'].float().clone()
        table[:, :, 0] /= self.stack_scale
        public = [obs['extra_info'].float()/self.stack_scale, table.flatten(1)]
        if self.betting:
            betting = obs['betting_info'].float().clone()
            betting[:, :6] /= self.stack_scale
            betting[:, 7] = torch.log1p(betting[:, 7])  # SPR can be large preflop.
            betting[:, 8] /= max(self.players-1, 1)
            betting[:, 9] /= 3
            public.append(betting)
        card = self.cards(obs['card_info'].float().permute(0,3,1,2)).flatten(1)
        action = self.actions(obs['action_info'].float().permute(0,3,1,2)).flatten(1)
        extra = self.public(torch.cat(public, dim=1))
        return self.fuse(torch.cat([card, action, extra], dim=1))


class NlHoldemStructuredNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self,obs_space,action_space,num_outputs,model_config,name)
        nn.Module.__init__(self)
        original = getattr(obs_space,'original_space',obs_space)
        if 'table_info' not in original.spaces:
            raise ValueError('Structured model requires public table_info')
        options = model_config.get('custom_model_config', {})
        scale = float(options.get('stack_scale',200.0))
        if not 0 < scale < float('inf'):
            raise ValueError('stack_scale must be finite and positive')
        self.relative_seats = options.get('relative_seats',True)
        self.policy_encoder = HoldemFeatureEncoder(original,scale)
        self.separate_critic = options.get('separate_critic',True)
        if self.separate_critic:
            self.value_encoder = HoldemFeatureEncoder(original,scale)
        self.policy_head = SlimFC(64,num_outputs,initializer=normc_initializer(0.01),activation_fn=None)
        self.value_head = SlimFC(64,1,initializer=normc_initializer(1.0),activation_fn=None)

    def forward(self,input_dict,state,seq_lens):
        obs = input_dict['obs']
        if self.relative_seats:
            obs = actor_relative_observation(obs)
        features = self.policy_encoder(obs)
        value_features = self.value_encoder(obs) if self.separate_critic else features
        self._value = self.value_head(value_features).squeeze(-1)
        logits = self.policy_head(features)
        return logits.masked_fill(obs['legal_moves'] <= 0,-1e9),state

    def value_function(self):
        return self._value
