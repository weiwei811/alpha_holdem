"""Device selection without platform-specific checkpoint formats."""
import torch


def resolve_device(requested='auto'):
    if requested not in ('auto', 'cpu', 'cuda', 'mps'):
        raise ValueError('device must be auto, cpu, cuda, or mps')
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if requested == 'auto':
        requested = 'cuda' if cuda else 'mps' if mps else 'cpu'
    if requested == 'cuda' and not cuda:
        raise ValueError('CUDA is unavailable. Install the CUDA PyTorch wheel or select --device cpu.')
    if requested == 'mps' and not mps:
        raise ValueError('MPS is unavailable. Use an Apple Silicon Mac with an MPS-enabled PyTorch build, or --device cpu.')
    return torch.device(requested)


def configure_training_device(conf, requested='auto', gpus=None):
    # Preserve the earlier --gpus 0/1 interface as an explicit device choice.
    if gpus is not None:
        if gpus not in (0, 1):
            raise ValueError('This entry point supports zero or one learner GPU')
        if gpus == 1:
            if requested not in ('auto', 'cuda'):
                raise ValueError('--gpus 1 requires --device cuda or auto')
            requested = 'cuda'
        elif requested == 'auto':
            requested = 'cpu'
        elif requested == 'cuda':
            raise ValueError('--gpus 0 conflicts with --device cuda')
    device = resolve_device(requested)
    conf['num_gpus'] = 1 if device.type == 'cuda' else 0
    conf['num_gpus_per_env_runner'] = 0
    conf.setdefault('env_config', {})['learner_device'] = device.type
    if device.type == 'mps':
        conf['simple_optimizer'] = True
        conf['_separate_vf_optimizer'] = False
    return device
