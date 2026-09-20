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
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # PyTorch uses NCHW. The input is NHWC from the env (because of old TF config).
        # We need to permute inside forward.
        # card_info: (batch, 4, 13, 6) -> permute -> (batch, 6, 4, 13)
        self.card_conv = nn.Sequential(
            ResBlock(6, 16, 3, 1),
            ResBlock(16, 32, 3, 2),
            ResBlock(32, 64, 3, 2),
        )
        
        # Output of card_conv:
        # Input: 6x4x13
        # Stride 1: 16x4x13
        # Stride 2: 32x2x7
        # Stride 2: 64x1x4
        # Flattened size: 64 * 1 * 4 = 256
        self.card_flat_size = 64 * 1 * 4
        
        # action_info: (batch, 8, 5, 81) -> permute -> (batch, 81, 8, 5)
        self.action_conv = nn.Sequential(
            ResBlock(81, 16, 3, 1),
            ResBlock(16, 32, 3, 2),
            ResBlock(32, 64, 3, 2),
        )
        
        # Output of action_conv:
        # Input: 81x8x5
        # Stride 1: 16x8x5
        # Stride 2: 32x4x3
        # Stride 2: 64x2x2
        # Flattened size: 64 * 2 * 2 = 256
        self.action_flat_size = 64 * 2 * 2
        
        self.extra_fc = nn.Sequential(
            SlimFC(6, 16, initializer=normc_initializer(0.01), activation_fn="relu")
        )
        
        # feature_fuse = 256 + 256 + 16 = 528
        self.fc = nn.Sequential(
            SlimFC(528, 256, initializer=normc_initializer(0.01), activation_fn="relu"),
            SlimFC(256, 128, initializer=normc_initializer(0.01), activation_fn="relu"),
            SlimFC(128, 64, initializer=normc_initializer(0.01), activation_fn="relu")
        )
        
        self.conv_fuse = SlimFC(64, 5, initializer=normc_initializer(0.01), activation_fn=None)
        self.value_out = SlimFC(64, 1, initializer=normc_initializer(0.01), activation_fn=None)

    def forward(self, input_dict, state, seq_lens):
        # Permute from NHWC to NCHW
        card_info = input_dict["obs"]["card_info"].float()
        card_info = card_info.permute(0, 3, 1, 2) # (B, 6, 4, 13)
        
        action_info = input_dict["obs"]["action_info"].float()
        action_info = action_info.permute(0, 3, 1, 2) # (B, 81, 8, 5)
        
        extra_info = input_dict["obs"]["extra_info"].float()
        
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
        # mask illegal moves with large negative number
        inf_mask = torch.clamp(torch.log(action_mask), min=torch.finfo(torch.float32).min)
        
        return logits + inf_mask, state

    def value_function(self):
        return self._value.reshape(-1)
