# scripts/smoke_test.py (replace with this)
import torch
import yaml

from encoders import build_encoder


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    x = torch.randn(2, 3, 224, 224)
    pair = (x, x.clone())

    cfg = load_cfg("configs/encoder_clip_b32.yaml")
    enc = build_encoder(cfg)
    out = enc.encode(pair)
    print(
        "[CLIP] left/right/fused:",
        out["left"].shape,
        out["right"].shape,
        out.get("fused").shape,
    )

    cfg2 = load_cfg("configs/encoder_pri3d_random.yaml")
    enc2 = build_encoder(cfg2)
    out2 = enc2.encode(pair)
    print(
        "[Pri3D] left/right/fused:",
        out2["left"].shape,
        out2["right"].shape,
        out2.get("fused").shape,
    )


if __name__ == "__main__":
    main()
