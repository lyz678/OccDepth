"""Shared Hydra, device and checkpoint handling for the supported entrypoints."""
import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path


ROOT = Path(__file__).resolve().parents[1]


def config_main(function):
    path = Path(os.environ.get('DATA_CONFIG', ROOT/'occdepth/config/semantic_kitti/multicam_flospdepth_crp_stereodepth_cascadecls_2080ti.yaml')).expanduser().resolve()
    return hydra.main(version_base='1.3', config_path=str(path.parent), config_name=path.stem)(function)


def trainer_device(config):
    count = int(config.n_gpus)
    if count < 0:
        raise ValueError('n_gpus must be >= 0 (0 selects CPU)')
    if count and (not torch.cuda.is_available() or torch.cuda.device_count() < count):
        raise RuntimeError(f'Requested {count} GPU(s), but only {torch.cuda.device_count()} available; use n_gpus=0 for CPU')
    return dict(accelerator='gpu' if count else 'cpu', devices=count or 1,
                strategy='ddp' if count > 1 else 'auto', sync_batchnorm=count > 1)


def inference_device(config):
    if int(config.n_gpus) > 1:
        raise ValueError("Prediction export is single-process; use n_gpus=1 (or 0 for CPU)")
    trainer_device(config)
    return torch.device('cuda:0' if config.n_gpus else 'cpu')


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def load_model(model_class, config, **kwargs):
    """Load a trusted local Lightning checkpoint, including legacy config metadata.

    This deliberately opts into pickle only at this explicit full-checkpoint
    boundary, never through a global environment override.
    """
    path = to_absolute_path(config.get('ckpt') or 'trained_models/occdepth.ckpt')
    if not Path(path).is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}; set ckpt=/path/to/trusted.ckpt')
    config = OmegaConf.merge(config, {'pretrained_backbone': False})
    return model_class.load_from_checkpoint(path, config=config, map_location='cpu',
                                            weights_only=False, strict=True, **kwargs)
