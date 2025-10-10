from pathlib import Path

import torch
import yaml

from encoders import build_encoder

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
