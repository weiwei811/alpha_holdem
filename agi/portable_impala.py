"""Pinned RLlib 2.49 adapter for experimental single-device MPS learning.

CUDA keeps RLlib's native GPU implementation. Remote rollout policies stay on
CPU. MPS uses the simple learner path, bypassing RLlib's CUDA-only context.
"""
import torch
from ray.rllib.algorithms.impala import Impala
from ray.rllib.algorithms.impala.impala_torch_policy import ImpalaTorchPolicy


def move_policy_to_device(policy, device):
    if len(policy.model_gpu_towers) != 1 or len(policy._optimizers) != 1:
        raise ValueError('MPS requires one model tower and one optimizer')
    model = policy.model
    old_parameters = list(model.parameters())
    model.to(device)
    new_parameters = list(model.parameters())
    replacement = dict(zip(old_parameters, new_parameters))
    for optimizer in policy._optimizers:
        for group in optimizer.param_groups:
            group['params'] = [replacement[p] for p in group['params']]
        if optimizer.state:
            raise ValueError('Move the policy before its first optimizer update')
    policy.device = torch.device(device)
    policy.devices = [policy.device]
    policy.model_gpu_towers = [model]
    policy.unwrapped_model = model
    policy._state_inputs = model.get_initial_state()
    policy.exploration.device = policy.device
    if hasattr(policy.exploration, 'random_exploration'):
        policy.exploration.random_exploration.device = policy.device


def single_device_gradients(policy, sample_batches):
    if len(sample_batches) != 1 or len(policy._optimizers) != 1:
        raise ValueError('MPS gradient calculation requires one batch and one optimizer')
    if policy.distributed_world_size:
        raise ValueError('Distributed MPS learning is not supported')
    optimizer = policy._optimizers[0]
    optimizer.zero_grad(set_to_none=True)
    loss = policy.loss(policy.model, policy.dist_class, sample_batches[0])
    if isinstance(loss, (tuple, list)):
        if len(loss) != 1:
            raise ValueError('MPS requires a single combined loss')
        loss = loss[0]
    if hasattr(policy.model, 'custom_loss'):
        loss = policy.model.custom_loss([loss], sample_batches[0])[0]
    loss.backward()
    info = {'allreduce_latency': 0.0}
    info.update(policy.extra_grad_process(optimizer, loss))
    return [([p.grad for p in policy.model.parameters()], info)]


class PortableImpalaTorchPolicy(ImpalaTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self._metal_learner = (config.get('worker_index', 0) == 0 and
                              config.get('env_config', {}).get('learner_device') == 'mps')
        if self._metal_learner:
            from agi.devices import resolve_device
            move_policy_to_device(self, resolve_device('mps'))

    def stats_fn(self, train_batch):
        stats = super().stats_fn(train_batch)
        # Scaled rollout diagnostics distinguish noisy poker returns from a
        # reward/termination data-flow fault. Avoid changing the native loss.
        rewards = torch.as_tensor(train_batch['rewards']).detach().float()
        terminals = torch.as_tensor(train_batch['terminateds']).detach().float()
        stats.update(reward_mean=float(rewards.mean().cpu()),
                     reward_abs_max=float(rewards.abs().max().cpu()),
                     terminal_fraction=float(terminals.mean().cpu()))
        if 'vf_preds' in train_batch:
            values = torch.as_tensor(train_batch['vf_preds']).detach().float()
            stats['sampled_value_std'] = float(values.std(unbiased=False).cpu())
        return stats

    def set_state(self, state):
        super().set_state(state)
        # Ray's old policy stack saves this counter but does not restore it.
        self.num_grad_updates = int(state.get('num_grad_updates', 0))

    def _multi_gpu_parallel_grad_calc(self, sample_batches):
        if getattr(self, '_metal_learner', False):
            return single_device_gradients(self, sample_batches)
        return super()._multi_gpu_parallel_grad_calc(sample_batches)


class PortableImpala(Impala):
    @classmethod
    def get_default_policy_class(cls, config):
        return PortableImpalaTorchPolicy


def configure_entropy(policy, coefficient):
    """Apply an intentional coefficient change after RLlib restores policy config."""
    import math
    from ray.rllib.policy.torch_mixins import EntropyCoeffSchedule
    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Entropy coefficient must be finite and nonnegative')
    policy.config['entropy_coeff'] = coefficient
    policy.config['entropy_coeff_schedule'] = None
    EntropyCoeffSchedule.__init__(policy, coefficient, None)
