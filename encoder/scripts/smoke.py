# scripts/smoke.py
import os, sys, argparse
from pathlib import Path
ROOT = os.path.dirname(os.path.dirname(__file__)); sys.path.insert(0, ROOT)

import torch
import yaml
from PIL import Image
from encoder import build_encoder
from encoder.transforms import build_image_transform, prepare_batch

CFG_DIR = Path(ROOT) / "configs"

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def pick_clip_cfg(use_pretrained: bool) -> str:
    if use_pretrained:
        p = CFG_DIR / "encoder_clip_b32_openai.yaml"
        if p.exists():
            return str(p)
        print("[warn] pretrained requested but encoder_clip_b32_openai.yaml not found; using stub config.")
    return str(CFG_DIR / "encoder_clip_b32.yaml")

def random_tensor_demo():
    print("=== Random tensor demo ===")
    x = torch.randn(2, 3, 224, 224)
    pair = (x, x.clone())

    enc = build_encoder(load_cfg(pick_clip_cfg(use_pretrained=False)))
    out = enc.encode(pair)
    print("[CLIP]   left/right/fused:", out["left"].shape, out["right"].shape, out.get("fused").shape)

    enc2 = build_encoder(load_cfg("configs/encoder_pri3d_random.yaml"))
    out2 = enc2.encode(pair)
    print("[Pri3D]  left/right/fused:", out2["left"].shape, out2["right"].shape, out2.get("fused").shape)

def image_demo(use_pretrained: bool):
    print("=== Image demo (with transforms) ===")
    enc = build_encoder(load_cfg(pick_clip_cfg(use_pretrained)))
    # Prefer the encoder's own transform if it exposes one (Step 3 ClipEncoder does)
    tfm = getattr(enc, "transform", None) or build_image_transform(kind="clip", size=224)


    # Replace with Image.open("path.jpg") to use real files
    left_img  = Image.new("RGB", (320, 240), color=(255, 0, 0))
    right_img = Image.new("RGB", (240, 320), color=(0, 255, 0))

    x_left  = prepare_batch(left_img,  transform=tfm)  # [1,3,224,224]
    x_right = prepare_batch(right_img, transform=tfm)  # [1,3,224,224]

    out = enc.encode((x_left, x_right))
    print("[CLIP]   left/right/fused:", out["left"].shape, out["right"].shape, out.get("fused").shape)

def pri3d_image_demo():
    print("=== Pri3D image demo (imagenet transforms) ===")
    tfm_im = build_image_transform(kind="imagenet", size=224)
    left  = Image.new("RGB", (320, 240), color=(30, 30, 200))
    right = Image.new("RGB", (240, 320), color=(30, 200, 30))
    x_left  = prepare_batch(left,  transform=tfm_im)
    x_right = prepare_batch(right, transform=tfm_im)
    enc_p = build_encoder(load_cfg("configs/encoder_pri3d_random.yaml"))
    out_p = enc_p.encode((x_left, x_right))
    print("[Pri3D-img] left/right/fused:", out_p["left"].shape, out_p["right"].shape, out_p.get("fused").shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", action="store_true", help="Use CLIP ViT-B/32 (openai) if config exists")
    args = ap.parse_args()

    random_tensor_demo()
    image_demo(use_pretrained=args.pretrained)
    pri3d_image_demo()

if __name__ == "__main__":
    main()
