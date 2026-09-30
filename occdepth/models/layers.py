"""Native layers preserving MMDetection 2.20 / MMCV 1.4 checkpoint names.

Behavior adapted from OpenMMLab's Apache-2.0 licensed implementations.
This is intentionally not a general replacement for MM registries.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = deepcopy(init_cfg)
        self._is_init = False

    def init_weights(self):
        if self._is_init:
            return
        for child in self.children():
            if hasattr(child, "init_weights"):
                child.init_weights()
        configs = self.init_cfg or []
        if isinstance(configs, dict):
            configs = [configs]
        for original in configs:
            cfg = dict(original)
            kind = cfg.pop("type")
            if kind == "Pretrained":
                path = cfg.pop("checkpoint")
                if cfg:
                    raise ValueError(f"Unsupported pretrained options: {cfg}")
                state = torch.load(path, map_location="cpu", weights_only=True)
                self.load_state_dict(state.get("state_dict", state), strict=True)
                continue
            layers = cfg.pop("layer", [])
            layers = [layers] if isinstance(layers, str) else layers
            if kind not in ("Kaiming", "Constant"):
                raise ValueError(f"Unsupported initialization: {kind}")
            allowed = {"val", "bias"} if kind == "Constant" else {"a", "mode", "nonlinearity", "distribution", "bias"}
            if set(cfg) - allowed:
                raise ValueError(f"Unsupported initialization options: {cfg}")
            for module in self.modules():
                if not any(base.__name__ in layers for base in type(module).__mro__):
                    continue
                if kind == "Constant":
                    nn.init.constant_(module.weight, cfg.get("val", 1))
                else:
                    distribution = cfg.get("distribution", "normal")
                    if distribution not in ("normal", "uniform"):
                        raise ValueError(f"Unsupported distribution: {distribution}")
                    initializer = getattr(nn.init, f"kaiming_{distribution}_")
                    initializer(module.weight, a=cfg.get("a", 0), mode=cfg.get("mode", "fan_out"), nonlinearity=cfg.get("nonlinearity", "relu"))
                if module.bias is not None:
                    nn.init.constant_(module.bias, cfg.get("bias", 0))
        self._is_init = True


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias="auto", conv_cfg=None,
                 norm_cfg=None, act_cfg=None):
        super().__init__()
        if conv_cfg not in (None, {}, {"type": "Conv2d"}, {"type": "Conv"}):
            raise ValueError(f"Unsupported convolution: {conv_cfg}")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups,
                              bias=(norm_cfg is None) if bias == "auto" else bias)
        self.norm_name = None
        if norm_cfg is not None:
            cfg = dict(norm_cfg)
            kind = cfg.pop("type")
            requires_grad = cfg.pop("requires_grad", True)
            if kind in ("BN", "BN2d", "SyncBN"):
                self.norm_name = "bn"
                cls = nn.SyncBatchNorm if kind == "SyncBN" else nn.BatchNorm2d
                norm = cls(out_channels, **cfg)
            elif kind == "GN":
                self.norm_name = "gn"
                norm = nn.GroupNorm(cfg.pop("num_groups"), out_channels, **cfg)
            else:
                raise ValueError(f"Unsupported normalization: {kind}")
            norm.requires_grad_(requires_grad)
            self.add_module(self.norm_name, norm)
        self.act_cfg = deepcopy(act_cfg)
        if act_cfg is not None:
            cfg = dict(act_cfg)
            kind = cfg.pop("type")
            if kind not in ("ReLU", "ReLU6", "LeakyReLU", "Sigmoid", "Hardsigmoid", "SiLU"):
                raise ValueError(f"Unsupported activation: {kind}")
            if kind != "Sigmoid":
                cfg.setdefault("inplace", True)
            self.activate = getattr(nn, kind)(**cfg)
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name) if self.norm_name else None

    def init_weights(self):
        leaky = self.act_cfg and self.act_cfg["type"] == "LeakyReLU"
        nn.init.kaiming_normal_(self.conv.weight,
                               a=self.act_cfg.get("negative_slope", 0.01) if leaky else 0,
                               mode="fan_out", nonlinearity="leaky_relu" if leaky else "relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        if self.norm is not None and self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def forward(self, x, activate=True, norm=True):
        x = self.conv(x)
        if norm and self.norm is not None:
            x = self.norm(x)
        if activate and self.act_cfg is not None:
            x = self.activate(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, with_cp=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # Original BasicBlock applies dilation only to conv1.
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.with_cp = with_cp

    def forward(self, x):
        def residual(x):
            identity = x if self.downsample is None else self.downsample(x)
            return self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + identity
        out = checkpoint(residual, x, use_reentrant=False) if self.with_cp and x.requires_grad else residual(x)
        return self.relu(out)
