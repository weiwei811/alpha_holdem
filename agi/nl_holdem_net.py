import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import normc_initializer, SlimFC

torch, nn = try_import_torch()

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResBlock, self).__init__()
        self.stride = stride
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.relu2 = nn.ReLU()
        
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        
        shortcut = self.shortcut(x)
        return self.relu2(y + shortcut)

class NlHoldemNet(TorchModelV2, nn.Module):
    conv_width = 16
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        original = getattr(obs_space, 'original_space', obs_space)
        action_shape = original['action_info'].shape
        players = original['extra_info'].shape[0]
        self.has_table_info = 'table_info' in original.spaces
        self.stack_scale = float(model_config.get('custom_model_config', {}).get('stack_scale', 200.0))
        if self.stack_scale <= 0:
            raise ValueError('stack_scale must be positive')
        width = self.conv_width
        final = width * 4
        self.card_conv = nn.Sequential(
            ResBlock(6, width, 3, 1),
            ResBlock(width, width * 2, 3, 2),
            ResBlock(width * 2, final, 3, 2),
        )
        
        self.action_conv = nn.Sequential(
            ResBlock(action_shape[-1], width, 3, 1),
            ResBlock(width, width * 2, 3, 2),
            ResBlock(width * 2, final, 3, 2),
        )
        
        self.extra_fc = nn.Sequential(
            SlimFC(players * (6 if self.has_table_info else 1), 16, initializer=normc_initializer(0.01), activation_fn="relu")
        )
        
        self.fc = nn.Sequential(
            SlimFC(final * (4 + ((action_shape[0] + 3) // 4) * ((action_shape[1] + 3) // 4)) + 16, 256, initializer=normc_initializer(0.01), activation_fn="relu"),
            SlimFC(256, 128, initializer=normc_initializer(0.01), activation_fn="relu"),
            SlimFC(128, 64, initializer=normc_initializer(0.01), activation_fn="relu")
        )
        
        self.conv_fuse = SlimFC(64, num_outputs, initializer=normc_initializer(0.01), activation_fn=None)
        self.value_out = SlimFC(64, 1, initializer=normc_initializer(0.01), activation_fn=None)

    def forward(self, input_dict, state, seq_lens):
        card_info = input_dict["obs"]["card_info"].float()
        card_info = card_info.permute(0, 3, 1, 2)
        
        action_info = input_dict["obs"]["action_info"].float()
        action_info = action_info.permute(0, 3, 1, 2)
        
        extra_info = input_dict["obs"]["extra_info"].float() / self.stack_scale
        if self.has_table_info:
            table = input_dict['obs']['table_info'].float().clone()
            table[:, :, 0] /= self.stack_scale
            extra_info = torch.cat([extra_info, table.flatten(1)], dim=-1)
        
        card_out = self.card_conv(card_info)
        card_out = card_out.reshape(card_out.shape[0], -1)
        
        action_out = self.action_conv(action_info)
        action_out = action_out.reshape(action_out.shape[0], -1)
        
        extra_out = self.extra_fc(extra_info)
        
        fuse = torch.cat([card_out, action_out, extra_out], dim=-1)
        fc_out = self.fc(fuse)
        
        logits = self.conv_fuse(fc_out)
        self._value = self.value_out(fc_out)
        
        action_mask = input_dict["obs"]["legal_moves"].float()
        # A finite sentinel avoids float32 overflow in rollout diagnostics.
        return logits.masked_fill(action_mask <= 0, -1e9), state

    def value_function(self):
        return self._value.reshape(-1)
