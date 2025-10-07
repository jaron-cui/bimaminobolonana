# Bimaminobolonana

## Setup

### Install Dependencies

```
conda env create -f environment.yaml
conda activate dev
pip install -r requirements.txt
```

## Quickstart (Python 3.10, venv)
```
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# CPU PyTorch (works everywhere)
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.* torchvision==0.19.*
pip install -r requirements.txt
pytest -q
```
