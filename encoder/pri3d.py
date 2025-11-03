from __future__ import annotations
from typing import Optional, Dict, Tuple
import warnings
import torch
import torch.nn as nn
from torchvision import models
from .base import MultiViewEncoder

_BACKBONES: Dict[str, Tuple[str, int]] = {
    "resnet18": ("resnet18", 512),
    "resnet34": ("resnet34", 512),
    "resnet50": ("resnet50", 2048),
}

def _build_trunk(variant: str) -> tuple[nn.Module, int]:
    if variant not in _BACKBONES:
        raise ValueError(f"Unknown Pri3D variant '{variant}'. Options: {list(_BACKBONES.keys())}")
    name, feat_dim = _BACKBONES[variant]
    ctor = getattr(models, name)
    net = ctor(weights=None)     # random init by default
    # keep the full ResNet so state_dict keys stay like 'conv1', 'layer1.*', ...
    net.fc = nn.Identity()       # disable classifier; forward() will output [B, feat_dim]
    return net, feat_dim

def _load_sd_file(path: str) -> Dict[str, torch.Tensor]:
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("state_dict", "model", "model_state", "net", "backbone"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    # strip common prefixes
    def strip(prefix: str, key: str) -> str:
        return key[len(prefix):] if key.startswith(prefix) else key
    cleaned = {}
    for k, v in sd.items():
        k = strip("module.", k)
        k = strip("backbone.", k)
        k = strip("encoder.", k)
        k = strip("model.", k)
        cleaned[k] = v
    return cleaned

def _load_into_trunk(trunk: nn.Sequential, sd: Dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    trunk_sd = trunk.state_dict()
    filtered = {k: v for k, v in sd.items() if k in trunk_sd and v.shape == trunk_sd[k].shape}
    missing = [k for k in trunk_sd.keys() if k not in filtered]
    unexpected = [k for k in sd.keys() if k not in filtered]
    trunk.load_state_dict(filtered, strict=False)
    return missing, unexpected

class Pri3DEncoder(MultiViewEncoder):
    """
    Pri3D-style encoder using a torchvision ResNet trunk.
    - variant: resnet18|34|50
    - pretrained: when True and ckpt_path is set, load Pri3D weights into the trunk
    - ckpt_path: path to Pri3D checkpoint (torchvision-format or convertible)
    - freeze: freeze trunk + projector when True
    - out_dim: projected feature size (default 512)
    - fuse: None | "mean" | "max" | "concat_mlp" | "gated" | "bilinear"
    """
    def __init__(
        self,
        variant: str = "resnet50",
        pretrained: bool = False,
        ckpt_path: Optional[str] = None,
        freeze: bool = False,
        out_dim: int = 512,
        fuse: Optional[str] = None,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__(out_dim=out_dim, fuse=fuse)
        self.variant = variant
        self.backbone, feat_dim = _build_trunk(variant)
        self.proj = nn.Identity() if feat_dim == out_dim else nn.Linear(feat_dim, out_dim, bias=False)

        # Optional: load Pri3D pretrained weights
        if pretrained:
            if not ckpt_path:
                warnings.warn("Pri3DEncoder: pretrained=True but no ckpt_path provided; using random init.")
            else:
                try:
                    sd = _load_sd_file(ckpt_path)
                    missing, unexpected = _load_into_trunk(self.backbone, sd)
                    matched = len(self.backbone.state_dict()) - len(missing)
                    warnings.warn(f"Pri3D ckpt loaded from '{ckpt_path}': matched={matched}, missing={len(missing)}, unexpected={len(unexpected)}")
                except Exception as e:
                    warnings.warn(f"Pri3DEncoder: failed to load checkpoint '{ckpt_path}': {e}")

        if device is not None:
            self.to(device)

        if freeze:
            # freeze trunk + projector; fusion (if any) remains trainable only if you manage it separately
            for m in (self.backbone, self.proj):
                for p in m.parameters():
                    p.requires_grad = False

        self.expected_preprocess = "imagenet"

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects ImageNet-normalized float32 [B,3,H,W].
        Returns [B, out_dim].
        """
        dev = next(self.backbone.parameters()).device
        x = x.to(device=dev, dtype=torch.float32)
        feats = self.backbone(x)      # [B, feat_dim] because fc is Identity()
        return self.proj(feats)       # [B, out_dim]
