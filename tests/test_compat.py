import copy
import importlib.util

import pytest
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock as TorchvisionBlock

from occdepth.models.layers import BasicBlock, ConvModule
from occdepth.models.mobilenet.mobilenet_v2 import MobileNetV2
from occdepth.models.mobilenet.inverted_residual import InvertedResidual
from occdepth.models.mobilenet.se_layer import SELayer
from occdepth.models.unet2d import UNet2D, MODEL_CHANNELS, NUM_FEATURES
from occdepth.runtime import trainer_device
from omegaconf import OmegaConf


torch.set_num_threads(2)


@pytest.mark.parametrize('training', [False, True])
def test_residual_matches_reference_outputs_and_gradients(training):
    # At the default settings used by OccDepth, torchvision's residual block
    # provides an independent reference with exactly the same state_dict.
    actual = BasicBlock(8, 8).train(training)
    reference = TorchvisionBlock(8, 8).train(training)
    reference.load_state_dict(actual.state_dict(), strict=True)
    x = torch.randn(2, 8, 9, 11, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a, b = actual(x), reference(y)
    torch.testing.assert_close(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key])


def test_conv_and_se_state_and_backward():
    conv = ConvModule(8, 8, 3, padding=1, groups=8,
                      norm_cfg={'type': 'BN'}, act_cfg={'type': 'ReLU6'})
    assert conv.conv.bias is None
    assert 'bn.running_mean' in conv.state_dict()
    plain = ConvModule(8, 4, 1, act_cfg={'type': 'Sigmoid'})
    assert plain.conv.bias is not None
    se = SELayer(8, ratio=2)
    x = torch.randn(2, 8, 7, 9, requires_grad=True)
    plain(se(conv(x))).mean().backward()
    assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):
        ConvModule(8, 8, 1, norm_cfg={'type': 'Unsupported'})


def test_mobilenet_freezing_initialization_and_checkpoint():
    model = MobileNetV2(frozen_stages=2, norm_eval=True, with_cp=True)
    model.init_weights()
    before = model.conv1.conv.weight.clone()
    model.init_weights()
    torch.testing.assert_close(before, model.conv1.conv.weight)
    model.train()
    assert not any(p.requires_grad for p in model.layer2.parameters())
    assert all(not m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    outputs = model(torch.randn(2, 3, 64, 64, requires_grad=True))
    assert [o.shape[1] for o in outputs] == [32, 64, 96, 1280]
    sum(o.mean() for o in outputs).backward()
    block = InvertedResidual(8, 8, 16, with_cp=False).eval()
    checked = copy.deepcopy(block)
    checked.with_cp = True
    a = torch.randn(2, 8, 8, 8, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    block(a).sum().backward()
    checked(b).sum().backward()
    torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize('name', list(MODEL_CHANNELS))
def test_backbone_feature_indices(name):
    model = UNet2D.build(out_feature=4, backbone_2d_name=name,
                        pretrained_backbone=False, return_up_feats=1).eval()
    with torch.no_grad():
        features = model.encoder(torch.randn(1, 3, 32, 64))
    assert [features[i].shape[1] for i in (4, 5, 6, 8)] == MODEL_CHANNELS[name][1:]
    assert features[11].shape[1] == NUM_FEATURES[name]
    with torch.no_grad():
        outputs = model.decoder(features)
    assert outputs['1_1'].shape == (1, 4, 32, 64)
    assert outputs['1_8'].shape[-2:] == (4, 8)


def test_device_config(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    assert trainer_device(OmegaConf.create({'n_gpus': 0}))['accelerator'] == 'cpu'
    assert trainer_device(OmegaConf.create({'n_gpus': 1}))['strategy'] == 'auto'
    assert trainer_device(OmegaConf.create({'n_gpus': 2}))['strategy'] == 'ddp'
    with pytest.raises(RuntimeError):
        trainer_device(OmegaConf.create({'n_gpus': 3}))
    assert importlib.util.find_spec('mmdet') is None
    assert importlib.util.find_spec('mmcv') is None


def test_invisible_voxels_have_finite_backward():
    from occdepth.models.SFA import SFA
    layer = SFA((2,2,2), 'kitti', 1)
    image = torch.randn(2,4,8,8,requires_grad=True)
    pixels = torch.zeros(2,8,1,2,dtype=torch.long)
    mask = torch.zeros(2,8,1,dtype=torch.bool)
    mask[:,0] = True
    out = layer(image,pixels,mask)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert torch.isfinite(image.grad).all()
