from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

Tensor = torch.Tensor
ImgOrPair = Union[Tensor, Tuple[Tensor, Tensor]]


# Lazy import to avoid circulars when fusion imports encoders, etc.
def _get_fuser(name: Optional[str], dim: int):
    if name is None:
        return None
    from encoder.fusion import fuse_mean, fuse_max, ConcatMLP, GatedFusion, BilinearFusion
    if name == "mean":
        return lambda a, b: fuse_mean(a, b)
    if name == "max":
        return lambda a, b: fuse_max(a, b)
    if name == "concat_mlp":
        return ConcatMLP(in_dim=dim, out_dim=dim)
    if name == "gated":
        return GatedFusion(dim)
    if name == "bilinear":
        return BilinearFusion(dim)
    raise ValueError(f"Unknown fuse option: {name}")


class MultiViewEncoder(nn.Module, ABC):
    """
    Base interface for single- and two-view encoders.
    Subclasses must implement forward_single(x) -> BxD features.
    """

    def __init__(self, out_dim: int = 512, fuse: Optional[str] = None) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.fuse_name = fuse
        # Module or callable:
        fuser = _get_fuser(fuse, self.out_dim)
        # If it's an nn.Module, register it; else keep as callable
        self.fuser_module = fuser if isinstance(fuser, nn.Module) else None
        self.fuser_fn = None if isinstance(fuser, nn.Module) else fuser

    @abstractmethod
    def forward_single(self, x: Tensor) -> Tensor:
        """Encode a single Bx3xHxW image tensor -> BxD."""
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, x: ImgOrPair) -> Dict[str, Tensor]:
        """
        Inference-friendly wrapper. Returns dict with keys:
        - 'left', 'right' (if a pair is given)
        - 'fused' if fuse is enabled
        """
        was_training = self.training
        self.eval()
        try:
            if isinstance(x, tuple):
                left, right = x
                f_left = self.forward_single(left)
                f_right = self.forward_single(right)
                out = {"left": f_left, "right": f_right}
                if self.fuse_name:
                    fused = self._fuse(f_left, f_right)
                    out["fused"] = fused
                return out
            else:
                f = self.forward_single(x)
                return {"left": f}
        finally:
            self.train(was_training)

    def _fuse(self, f_left: Tensor, f_right: Tensor) -> Tensor:
        if self.fuser_module is not None:
            return self.fuser_module(f_left, f_right)
        if self.fuser_fn is not None:
            return self.fuser_fn(f_left, f_right)
        raise RuntimeError("Fusion requested but no fuser is set.")
