import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent.absolute()))
os.chdir("..") # change to repo root dir

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from encoder import build_encoder
from encoder.transforms import build_image_transform, prepare_batch
from train.dataset import TensorBimanualObs, TensorBimanualAction
from train.trainer import BimanualActor,BCTrainer
from train.dataset import BimanualDataset
from train.dataset import generate_bimanual_dataset
from encoder.transforms import build_image_transform
import yaml

from torch.utils.data import DataLoader
from train.dataset import BimanualDataset
from train.trainer import BCTrainer
from train.train_utils import Logs
import os

print("Environment ready.")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generating Training Data from Privilidged Policy
dataset_dir_path = Path('bc-train-data-test')
generate_bimanual_dataset(
  save_dir=dataset_dir_path,
  total_sample_count=10000,
  max_steps_per_rollout=600,
  skip_frames=2,
  camera_dims=(128, 128),
  resume=True
)

dataset = BimanualDataset(Path("train/bc-train-data-test"))
obs, act = dataset[0]

class BimanualPri3DActor(nn.Module):
    def __init__(self, encoder_cfg_path="configs/encoder_pri3d_pretrained.yaml",
                 state_dim=32, hidden_dim=512, action_dim=14,
                 freeze_encoder=False):   
        super().__init__()

        # --- Load Pri3D encoder ---
        with open(encoder_cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.encoder = build_encoder(cfg)

        if freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            self.encoder.train()
            for p in self.encoder.parameters():
                p.requires_grad = True

        # --- Fusion MLP ---
        self.mlp = nn.Sequential(
            nn.Linear(512 + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, obs: TensorBimanualObs) -> TensorBimanualAction:
        left_imgs  = obs.visual[:, 0].permute(0, 3, 1, 2)
        right_imgs = obs.visual[:, 1].permute(0, 3, 1, 2)

        feats = self.encoder.encode((left_imgs, right_imgs))
        fused_feat = feats["fused"]

        state_feat = torch.cat([obs.qpos.array, obs.qvel.array], dim=-1)

        x = torch.cat([fused_feat, state_feat], dim=-1)
        out = self.mlp(x)
        return TensorBimanualAction(out)


BATCH_SIZE = 64
NUM_EPOCHS = 100
dataset_dir_path = "train/bc-train-data-test"
dataset = BimanualDataset(dataset_dir_path)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=BimanualDataset.collate_fn)

os.makedirs("out/training-pri3d", exist_ok=True)
logs = Logs("out/training-pri3d")
job = logs.create_new_job(tag="pri3d-bc")

model = BimanualPri3DActor(freeze_encoder=False).cuda()


optimizer = torch.optim.Adam([
    {"params": model.encoder.parameters(), "lr": 1e-5},
    {"params": model.mlp.parameters(), "lr": 1e-4},
])

trainer = BCTrainer(dataloader, checkpoint_frequency=5, job=job)
trainer.train(model, num_epochs=NUM_EPOCHS, optimizer=optimizer)
