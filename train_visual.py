import glob
import os
import pathlib
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from robot.sim import JOINT_OBSERVATION_SIZE
from train.dataset import BimanualDataset, FilteredDataset, TensorBimanualAction, TensorBimanualObs
from train.train_utils import Logs
from policy.act.policy_visual import VisualOnlyMLPPolicy

# ============================================================
#                CONFIGURATION
# ============================================================

CONFIG_PATH = "configs/policy_visual.yaml"  
DATASET_PATH = "/mnt/data/pickup-randomization0_05"
SAVE_DIR = "out/visual_only_mlp_bc"
DEVICE = 'cuda'
RELATIVE_ACTIONS = True

print("Using device:", DEVICE)  

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
    # proprio_dim=1
).to(DEVICE)

print("VisualOnlyMLPPolicy created.")

# ============================================================
#                   LOAD DATASET
# ============================================================

datasets = [BimanualDataset(subfolder) for subfolder in glob.glob(str(pathlib.Path(DATASET_PATH) / '*'))]
# datasets = [BimanualDataset('D:/bimaminobolonana/pickup-randomization0_05')]
means = []
vars = []
weights = []
for dataset in datasets:
    if RELATIVE_ACTIONS:
        visual_shape = (1, 2, dataset.metadata.camera_height, dataset.metadata.camera_width, 3)
        visual_size = np.array(visual_shape).prod()
        qpos = dataset._observation_array[:, visual_size:visual_size + JOINT_OBSERVATION_SIZE].clone()
        approx_actions = torch.cat([qpos[:, :7], qpos[:, 8:15]], dim=1)
        # # Set grippers as defined in to_approximate_action()
        # approx_actions[:, 6]  = qpos[:, 6] * 10    # left gripper
        # approx_actions[:, 13] = qpos[:, 14] * 10   # right gripper

        relative_actions = dataset._action_array - approx_actions
        relative_actions[:, 6] = dataset._action_array[:, 6]
        relative_actions[:, 13] = dataset._action_array[:, 13]
        means.append(relative_actions.mean(dim=0))
        vars.append(relative_actions.std(dim=0))
        weights.append(len(dataset))
    else:
        means.append(dataset._action_array.mean(dim=0))
        vars.append(dataset._action_array.std(dim=0))
        weights.append(len(dataset))
weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)
weights /= weights.sum()

combined_mean = (torch.stack(means) * weights).sum(dim=0)
# set the mean and std buffers for action normalization to standard 0-centered normal distribution
policy.action_mean[:] = combined_mean
policy.action_std[:] = ((torch.stack(vars) * weights).sum(dim=0) + ((torch.stack(means) - combined_mean).square() * weights).sum(dim=0)).sqrt()
print(policy.action_mean, policy.action_std)

dataset = torch.utils.data.ConcatDataset(datasets)
# print(len(dataset))
# def filter_nonmoving(sample: tuple[TensorBimanualObs, TensorBimanualAction]) -> bool:
#     obs = sample[0].qpos.to_approximate_action()
#     left_gripper_obs = obs.array[0, 6]
#     action = sample[1]
#     return (obs.array - action.array).abs().sum().item() > 0.2
    
# dataset = FilteredDataset(dataset, keep=filter_nonmoving)
# print(len(dataset))

dataloader = DataLoader(
    dataset,
    batch_size=512,#cfg["batch_size"],
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
NUM_EPOCHS = 20#cfg.get("num_epochs", 100)
CHECKPOINT_FREQ = 1#cfg.get("checkpoint_frequency", 10)


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
        obs = obs.to(DEVICE)

        obs.visual = obs.visual.to(DEVICE)
        action = action.to(DEVICE)
        if RELATIVE_ACTIONS:
            mask = torch.ones_like(action.array, dtype=torch.bool)
            # all actions are relative, except for gripper values, which are absolute
            mask[:, 6] = 0
            mask[:, 13] = 0
            prev_action = obs.qpos.to_approximate_action().to(DEVICE)
            action.array[mask] -= prev_action.array[mask]
            # action.array[:, 6] /= 0.1
            # action.array[:, 13] /= 0.1

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
        lambda_penalty = 0.0

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