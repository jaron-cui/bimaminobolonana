# Bimaminobolonana

## Setup

### Install Dependencies

```bash
conda env create -f environment.yaml
conda activate dev
pip install -r requirements.txt
```

## Quickstart (Python 3.10, venv)
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# CPU PyTorch (works everywhere)
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1
pip install -r requirements.txt
pytest -q
```

## Encoder skeleton quickstart
```python
from encoders import build_encoder
from omegaconf import OmegaConf  # or yaml.safe_load
enc = build_encoder(OmegaConf.load("configs/encoder_clip_b32.yaml"))
feats = enc.encode((images_left, images_right))["fused"]  # B x 512
```

## Transforms (input preprocessing)

Use our helpers to get correctly sized/normalized tensors for each encoder.

```python
from encoders.transforms import build_image_transform, prepare_batch

# CLIP-style preprocessing (use for clip_vit)
tfm = build_image_transform(kind="clip", size=224)
x_left  = prepare_batch(left_pil_image,  transform=tfm)   # [1,3,224,224]
x_right = prepare_batch(right_pil_image, transform=tfm)   # [1,3,224,224]

# ImageNet-style preprocessing (use for pri3d)
tfm_im = build_image_transform(kind="imagenet", size=224)
x = prepare_batch([img1, img2, img3], transform=tfm_im)   # [B,3,224,224]
```

## CLIP encoder (ViT-B/32)

We ship a real CLIP image encoder via `open-clip`. By default the config uses no pretrained weights (CI-safe); use the `openai` config for real features.

**Instantiate**
```python
from encoders import build_encoder
from omegaconf import OmegaConf
enc = build_encoder(OmegaConf.load("configs/encoder_clip_b32.yaml"))             # stub (no weights)
# enc = build_encoder(OmegaConf.load("configs/encoder_clip_b32_openai.yaml"))   # pretrained
```
