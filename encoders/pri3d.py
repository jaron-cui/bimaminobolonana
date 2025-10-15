from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional
from torchvision import models

from .base import MultiViewEncoder

_BACKBONES = {
    "resnet18":  ("resnet18", 512),
    "resnet34":  ("resnet34", 512),
    "resnet50":  ("resnet50", 2048),
}

def _build_backbone(variant: str) -> tuple[nn.Module, int]:
    if variant not in _BACKBONES:
        raise ValueError(f"Unknown Pri3D variant '{variant}'. Options: {list(_BACKBONES.keys())}")
    name, feat_dim = _BACKBONES[variant]
    constructor = getattr(models, name)
    # Random init (capacity-matched control)
    net = constructor(weights=None)
    # Drop classifier head; keep conv trunk that outputs CxHxW features
    # (for ResNet, it's everything except the final 'fc')
    modules = list(net.children())[:-1]  # up to global avgpool
    backbone = nn.Sequential(*modules)   # outputs [B, C, 1, 1]
    return backbone, feat_dim

class Pri3DEncoder(MultiViewEncoder):
    """
    Capacity-matched Pri3D-style encoder using a ResNet backbone.
    - variant: resnet18|34|50
    - pretrained: (unused for now) kept for API compatibility; set False
    - freeze: freeze backbone + projector when True
    - out_dim: projector output dim (default 512 to match project)
    - fuse: None | "mean" | "max" | "concat_mlp"
    """
    def __init__(
        self,
        variant: str = "resnet50",
        pretrained: bool = False,   # kept for config symmetry; ignored (weights=None)
        freeze: bool = False,
        out_dim: int = 512,
        fuse: Optional[str] = None,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__(out_dim=out_dim, fuse=fuse)
        self.variant = variant
        self.backbone, feat_dim = _build_backbone(variant)

        # Projector to a consistent feature size
        self.proj = nn.Identity() if feat_dim == out_dim else nn.Linear(feat_dim, out_dim, bias=False)

        if device is not None:
            self.to(device)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        # Document expected preprocessing (ImageNet stats)
        self.expected_preprocess = "imagenet"

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects ImageNet-normalized float32 tensor [B,3,H,W].
        Returns [B, out_dim].
        """
        dev = next(self.backbone.parameters()).device
        x = x.to(device=dev, dtype=torch.float32)

        feats = self.backbone(x)          # [B, C, 1, 1] (global avgpool already applied)
        feats = feats.flatten(1)          # -> [B, C]
        feats = self.proj(feats)          # -> [B, out_dim]
        return feats
