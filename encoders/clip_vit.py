from __future__ import annotations

import torch
import torch.nn as nn

from .base import MultiViewEncoder


class ClipEncoder(MultiViewEncoder):
    """
    Placeholder CLIP encoder.
    Will replace this with real open-clip / CLIP loading.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        out_dim: int = 512,
        freeze: bool = True,
        fuse=None,
    ):
        super().__init__(out_dim=out_dim, fuse=fuse)
        self.model_name = model_name
        self.freeze = freeze
        # Dummy feature projector to the requested out_dim:
        self._proj = nn.Identity()

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        # Emit zeros as stand-in features for now:
        return torch.zeros(b, self.out_dim, device=x.device, dtype=x.dtype)
