from __future__ import annotations

import torch
import torch.nn as nn

from .base import MultiViewEncoder


class Pri3DEncoder(MultiViewEncoder):
    """
    Placeholder Pri3D encoder (random or pretrained in later steps).
    """

    def __init__(
        self,
        variant: str = "resnet50",
        pretrained: bool = False,
        freeze: bool = False,
        out_dim: int = 512,
        fuse=None,
    ):
        super().__init__(out_dim=out_dim, fuse=fuse)
        self.variant = variant
        self.pretrained = pretrained
        self.freeze = freeze
        self._proj = nn.Identity()
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.zeros(b, self.out_dim, device=x.device, dtype=x.dtype)
