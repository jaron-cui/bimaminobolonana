from __future__ import annotations

from typing import Any, Dict

from .base import MultiViewEncoder
from .clip_vit import ClipEncoder
from .pri3d import Pri3DEncoder

_REGISTRY = {
    "clip_vit": ClipEncoder,
    "pri3d": Pri3DEncoder,
}


def build_encoder(cfg: Dict[str, Any] | Any) -> MultiViewEncoder:
    """
    Build an encoder from a simple dict or an OmegaConf object.
    Expected keys for CLIP:
        name: "clip_vit"
        model_name: "ViT-B-32"
        out_dim: 512
        freeze: true
        fuse: null|mean|max|concat_mlp
    Expected keys for Pri3D:
        name: "pri3d"
        variant: "resnet50"
        pretrained: false|true
        freeze: false|true
        out_dim: 512
        fuse: ...
    """
    if cfg is None:
        raise ValueError(
            "Config is None. Did you forget to populate the YAML file or pass the right path?"
        )
    # Accept OmegaConf or dict
    try:
        # OmegaConf-like
        name = cfg.name
    except AttributeError:
        name = cfg.get("name")

    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown encoder name: {name}. Options: {list(_REGISTRY.keys())}"
        )

    cls = _REGISTRY[name]

    if name == "clip_vit":
        kwargs = {
            "model_name": getattr(cfg, "model_name", None) or cfg.get("model_name", "ViT-B-32"),
            "pretrained": getattr(cfg, "pretrained", None) if hasattr(cfg, "pretrained") else cfg.get("pretrained", None),
            "out_dim": int(getattr(cfg, "out_dim", None) or cfg.get("out_dim", 512)),
            "freeze": bool(getattr(cfg, "freeze", None) if hasattr(cfg, "freeze") else cfg.get("freeze", True)),
            "fuse": getattr(cfg, "fuse", None) if hasattr(cfg, "fuse") else cfg.get("fuse", None),
        }
        return cls(**kwargs)

    if name == "pri3d":
        kwargs = {
            "variant": getattr(cfg, "variant", None) or cfg.get("variant", "resnet50"),
            "pretrained": bool(
                getattr(cfg, "pretrained", None)
                if hasattr(cfg, "pretrained")
                else cfg.get("pretrained", False)
            ),
            "ckpt_path": getattr(cfg, "ckpt_path", None) or cfg.get("ckpt_path", None),  
            "freeze": bool(
                getattr(cfg, "freeze", None)
                if hasattr(cfg, "freeze")
                else cfg.get("freeze", False)
            ),
            "out_dim": int(getattr(cfg, "out_dim", None) or cfg.get("out_dim", 512)),
            "fuse": (
                getattr(cfg, "fuse", None)
                if hasattr(cfg, "fuse")
                else cfg.get("fuse", None)
            ),
        }
        return cls(**kwargs)

    raise AssertionError("Unreachable")
