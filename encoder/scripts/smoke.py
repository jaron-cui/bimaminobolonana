# scripts/smoke.py
import os, sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # …/bimaminobolonana
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from PIL import Image
from encoder import build_encoder
from encoder.transforms import build_image_transform, prepare_batch

CFG_DIR = Path(ROOT) / "configs"

def count_trainable(m) -> int:
    return sum(p.numel() for p in m.parameters() if getattr(p, "requires_grad", False))

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def pick_clip_cfg(use_pretrained: bool = True, rn50: bool = False) -> str:
    if rn50:
        p = CFG_DIR / "encoder_clip_rn50_openai.yaml"
        if p.exists():
            return str(p)
        print("[warn] CLIP RN50 requested but encoder_clip_rn50_openai.yaml not found; falling back.")
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
    print("Trainable params [CLIP]:", count_trainable(enc))
    out = enc.encode(pair)
    print("[CLIP]   left/right/fused:", out["left"].shape, out["right"].shape, out.get("fused").shape)

    enc2 = build_encoder(load_cfg("configs/encoder_pri3d_random.yaml"))
    print("Trainable params [Pri3D-random]:", count_trainable(enc2))
    out2 = enc2.encode(pair)
    print("[Pri3D]  left/right/fused:", out2["left"].shape, out2["right"].shape, out2.get("fused").shape)

def clip_image_demo(use_pretrained: bool, rn50: bool):
    print("=== Image demo (with transforms) ===")
    enc = build_encoder(load_cfg(pick_clip_cfg(use_pretrained, rn50)))
    enc = build_encoder(load_cfg(pick_clip_cfg(use_pretrained)))
    print("Trainable params [CLIP image demo]:", count_trainable(enc))
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
    print("Trainable params [Pri3D-img]:", count_trainable(enc_p))
    out_p = enc_p.encode((x_left, x_right))
    print("[Pri3D-img] left/right/fused:", out_p["left"].shape, out_p["right"].shape, out_p.get("fused").shape)

def pri3d_ckpt_demo(ckpt_override: str | None):
    print("=== Pri3D pretrained demo (imagenet transforms) ===")
    cfg_path = CFG_DIR / "encoder_pri3d_pretrained.yaml"
    if not cfg_path.exists():
        print("[Pri3D-ckpt] Skipped: configs/encoder_pri3d_pretrained.yaml not found.")
        return

    cfg = load_cfg(str(cfg_path))
    if ckpt_override:
        cfg["ckpt_path"] = ckpt_override

    ckpt = cfg.get("ckpt_path")
    if not ckpt or not Path(ckpt).expanduser().exists():
        print(f"[Pri3D-ckpt] Skipped: checkpoint path missing or not found: {ckpt!r}")
        return

    enc = build_encoder(cfg)
    print("Trainable params [Pri3D-ckpt]:", count_trainable(enc))
    tfm = build_image_transform(kind="imagenet", size=224)
    left  = Image.new("RGB", (320, 240), color=(30, 30, 200))
    right = Image.new("RGB", (240, 320), color=(30, 200, 30))
    x_left  = prepare_batch(left,  transform=tfm)
    x_right = prepare_batch(right, transform=tfm)
    out = enc.encode((x_left, x_right))
    print("[Pri3D-ckpt] left/right/fused:", out["left"].shape, out["right"].shape, out.get("fused").shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", action="store_true", help="Use CLIP ViT-B/32 (openai) if config exists")
    ap.add_argument("--pri3d", action="store_true", help="Run Pri3D pretrained demo")
    ap.add_argument("--ckpt", type=str, default=None, help="Override Pri3D checkpoint path")
    ap.add_argument("--clip-rn50", action="store_true", help="Use CLIP RN50 (openai) config if present")
    args = ap.parse_args()

    random_tensor_demo()
    clip_image_demo(use_pretrained=args.pretrained, rn50=args.clip_rn50)
    pri3d_image_demo()
    if args.pri3d:
        pri3d_ckpt_demo(args.ckpt)

if __name__ == "__main__":
    main()
