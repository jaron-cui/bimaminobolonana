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
