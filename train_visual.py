import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.dataset import BimanualDataset, TensorBimanualAction
from train.train_utils import Logs
from policy.act.policy_visual import VisualOnlyMLPPolicy

# ============================================================
#                CONFIGURATION
# ============================================================

CONFIG_PATH = "configs/policy_visual.yaml"  
DATASET_PATH = "/mnt/data/simple-pickup-no-randomization"
SAVE_DIR = "out/visual_only_mlp_bc"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
#               LOAD CONFIG FILE
# ============================================================

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

print("Loaded config:", CONFIG_PATH)


# ============================================================
#            BUILD VISUAL-ONLY MLP POLICY
# ============================================================

policy = VisualOnlyMLPPolicy(
    encoder_cfg=cfg["encoder"],
    hidden_dim=cfg.get("hidden_dim", 512),
).to(DEVICE)

print("VisualOnlyMLPPolicy created.")

# ============================================================
#                   LOAD DATASET
# ============================================================

dataset = BimanualDataset(DATASET_PATH)
dataloader = DataLoader(
    dataset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    collate_fn=BimanualDataset.collate_fn,
)

print(f"Loaded dataset: {DATASET_PATH}, size={len(dataset)}")

# ============================================================
#                   OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    policy.parameters(),
    lr=cfg.get("lr", 1e-4),
)

USE_L1 = cfg.get("action_loss", "l1") == "l1"
NUM_EPOCHS = cfg.get("num_epochs", 100)
CHECKPOINT_FREQ = cfg.get("checkpoint_frequency", 10)


# ============================================================
#               TRAINING LOOP
# ============================================================

logs = Logs(SAVE_DIR)
job = logs.create_new_job(tag="visual-only-mlp-bc")

print("\n==================== TRAINING START ====================\n")

for epoch in range(NUM_EPOCHS):
    policy.train()
    epoch_loss = 0.0

    for obs, action in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        # action: TensorBimanualAction
        # print(type(obs.visual))
        # print(obs.visual)
        # print(obs.visual.array.shape)
        # break

        action = action.to(DEVICE)

        pred_action: TensorBimanualAction = policy(obs)

        if USE_L1:
            loss_action = nn.L1Loss()(pred_action.array, action.array)
        else:
            loss_action = nn.MSELoss()(pred_action.array, action.array)

        loss = loss_action

        qpos = obs.qpos.array.to(DEVICE)  # (B, 16)

        # Build batched approximate action (B, 14)
        approx_actions = torch.cat(
            [
                qpos[:, :7],     # left arm joints
                qpos[:, 8:15],   # right arm joints
            ],
            dim=1
        )

        # Set grippers as defined in to_approximate_action()
        approx_actions[:, 6]  = qpos[:, 6] * 10    # left gripper
        approx_actions[:, 13] = qpos[:, 14] * 10   # right gripper

        # Distance between predicted action and approximate (stay-still) action
        dist = (pred_action.array - approx_actions).abs().sum(dim=1)

        # Penalize being too close (i.e., too still)
        threshold = 0.05
        penalty = torch.relu(threshold - dist).mean()

        # Weight of the penalty
        lambda_penalty = 0.1

        # Add to loss
        loss = loss + lambda_penalty * penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"[Epoch {epoch+1}] Loss = {avg_loss:.6f}")

    # job.log_scalar("loss", avg_loss, epoch + 1)

    # Save checkpoint
    if (epoch + 1) % CHECKPOINT_FREQ == 0:
        ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pt")
        torch.save(policy.state_dict(), ckpt_path)
        print("Checkpoint saved:", ckpt_path)

print("\n==================== TRAINING COMPLETE ====================\n")