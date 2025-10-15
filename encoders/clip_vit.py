from __future__ import annotations

import torch
import torch.nn as nn
import open_clip
from .base import MultiViewEncoder


class ClipEncoder(MultiViewEncoder):
    """
    CLIP ViT image encoder backed by open-clip.
    - model_name: e.g. "ViT-B-32", "ViT-L-14"
    - pretrained: e.g. "openai", "laion2b_s34b_b79k", or None (random init)
    - out_dim: projector output dim (kept at 512 for project consistency)
    - freeze: if True, no grads on encoder or projector
    - fuse: None | "mean" | "max" | "concat_mlp"
    - use_model_preprocess: if True, expose CLIP's own transform via .transform
      (We can still use encoders.transforms for a unified pipeline.)
    """
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str | None = None,   # use "openai" for real weights
        out_dim: int = 512,
        freeze: bool = True,
        fuse: str | None = None,
        device: str | torch.device | None = None,
        use_model_preprocess: bool = True,
    ):
        super().__init__(out_dim=out_dim, fuse=fuse)

        # Build model + (optional) CLIP-native preprocess
        self.model, self.__, self._model_tfm = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()
        if device is not None:
            self.model.to(device)

        # Figure out CLIP's native embed dim (e.g., 512 for ViT-B/32 "openai")
        with torch.no_grad():
            # open-clip exposes this attribute in visual backbone
            native_dim = getattr(getattr(self.model, "visual", self.model), "output_dim", None)
            if native_dim is None:
                # Fallback: many open-clip models project to text embed dim
                native_dim = getattr(self.model, "embed_dim", None)
            if native_dim is None:
                # Last resort: assume 512 (works for ViT-B/32 "openai")
                native_dim = 512
        self.native_dim = int(native_dim)

        # Projector to a consistent out_dim for the project
        self.proj = nn.Identity() if self.native_dim == self.out_dim else nn.Linear(self.native_dim, self.out_dim, bias=False)

        # Freezing policy
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.proj.parameters():
                p.requires_grad = False

        # Optional: expose CLIP's own transform
        self.use_model_preprocess = bool(use_model_preprocess)
        self.transform = self._model_tfm if use_model_preprocess else None

    @torch.no_grad()
    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects x as a normalized tensor Bx3xHxW.
        If you set use_model_preprocess=True and want to pass PIL/numpy,
        call self.transform() first or use encoders.transforms.
        """
        # open-clip.encode_image expects float tensor on the same device
        dev = next(self.model.parameters()).device
        x = x.to(device=dev, dtype=torch.float32)
        feats = self.model.encode_image(x)         # [B, native_dim]
        feats = self.proj(feats)                   # [B, out_dim]
        return feats
