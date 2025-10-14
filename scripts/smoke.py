# scripts/smoke.py
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__)); sys.path.insert(0, ROOT)

import torch
import yaml
from PIL import Image
from encoders import build_encoder
from encoders.transforms import build_image_transform, prepare_batch

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def random_tensor_demo():
    print("=== Random tensor demo ===")
    x = torch.randn(2, 3, 224, 224)
    pair = (x, x.clone())

    enc = build_encoder(load_cfg("configs/encoder_clip_b32.yaml"))
    out = enc.encode(pair)
    print("[CLIP]   left/right/fused:", out["left"].shape, out["right"].shape, out.get("fused").shape)

    enc2 = build_encoder(load_cfg("configs/encoder_pri3d_random.yaml"))
    out2 = enc2.encode(pair)
    print("[Pri3D]  left/right/fused:", out2["left"].shape, out2["right"].shape, out2.get("fused").shape)

def image_demo():
    print("=== Image demo (with transforms) ===")
    # Build the right transform (CLIP here; use 'imagenet' for Pri3D)
    tfm = build_image_transform(kind="clip", size=224)

    # If you have real files, replace these with Image.open("path.jpg")
    left_img  = Image.new("RGB", (320, 240), color=(255, 0, 0))
    right_img = Image.new("RGB", (240, 320), color=(0, 255, 0))

    x_left  = prepare_batch(left_img,  transform=tfm)  # [1,3,224,224]
    x_right = prepare_batch(right_img, transform=tfm)  # [1,3,224,224]

    enc = build_encoder(load_cfg("configs/encoder_clip_b32.yaml"))
    out = enc.encode((x_left, x_right))
    print("[CLIP]   left/right/fused:", out["left"].shape, out["right"].shape, out.get("fused").shape)

def main():
    random_tensor_demo()
    image_demo()

if __name__ == "__main__":
    main()
