from pathlib import Path

import torch
import yaml

from encoder import build_encoder

ROOT = Path(__file__).resolve().parents[1]


def _load(rel_path: str):
    with open(ROOT / rel_path, "r") as f:
        return yaml.safe_load(f)


def test_clip_registry_and_shapes():
    cfg = _load("configs/encoder_clip_b32.yaml")
    enc = build_encoder(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = enc.encode((x, x))
    assert out["left"].shape == (2, cfg["out_dim"])
    assert out["right"].shape == (2, cfg["out_dim"])
    assert out["fused"].shape == (2, cfg["out_dim"])


def test_pri3d_registry_and_shapes():
    cfg = _load("configs/encoder_pri3d_random.yaml")
    enc = build_encoder(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = enc.encode((x, x))
    assert out["left"].shape == (2, cfg["out_dim"])
    assert out["right"].shape == (2, cfg["out_dim"])
    assert out["fused"].shape == (2, cfg["out_dim"])


def test_clip_freezing_flag():
    from encoder import build_encoder
    cfg = {"name": "clip_vit", "model_name": "ViT-B-32", "pretrained": None, "freeze": True, "out_dim": 512}
    enc = build_encoder(cfg)
    # projector and backbone should be frozen
    assert all(not p.requires_grad for p in enc.parameters())


def test_pri3d_freezing_flag():
    cfg = {"name": "pri3d", "variant": "resnet50", "pretrained": False, "freeze": True, "out_dim": 512}
    from encoder import build_encoder
    enc = build_encoder(cfg)
    assert all(not p.requires_grad for p in enc.parameters())
