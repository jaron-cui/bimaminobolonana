"""
Visualize ACT (Action Chunking Transformer) Policy Rollouts

This script loads a trained ACT checkpoint and runs a rollout with visualization,
saving videos of the left and right camera views.

Usage:
    python inference_ACT.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from collections import deque
from omegaconf import OmegaConf

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

from policy.act import build_act_policy
from robot.sim import BimanualSim, BimanualObs, BimanualAction, randomize_block_position
from robot.visualize import save_frames_to_video
from train.dataset import TensorBimanualObs

current_file = Path(__file__).resolve()
project_root = current_file.parent

# =============== CONFIG ==================
CHECKPOINT = "runs/act_policy/150.pt"  # Path to trained ACT checkpoint
CONFIG_PATH = "configs/policy_act.yaml"  # Path to ACT config
MAX_STEPS = 600
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============== LOAD MODEL ==================
print(f"Loading config from: {CONFIG_PATH}")
config = OmegaConf.load(CONFIG_PATH)
print(f"Config: {config.name}")
print(f"  - Chunk size: {config.chunk_size}")
print(f"  - Temporal context: {config.temporal_context}")
print(f"  - Encoder: {config.encoder.name}")
print(f"  - Image size: {config.image_size}")
print()

print("Building ACT policy...")
model = build_act_policy(config).to(DEVICE)
model.eval()

print(f"Loading checkpoint from: {CHECKPOINT}")
state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state_dict)
print("✓ Checkpoint loaded successfully!")
print()

# =============== ACT POLICY WRAPPER ==================
class ACTPolicyWrapper:
    """
    Wrapper for ACT policy that handles temporal context and action chunking.
    """
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.temporal_context = config.temporal_context
        self.chunk_size = config.chunk_size
        self.ensemble_window = config.get('ensemble_window', 10) if config.get('temporal_ensemble', False) else 1

        # Temporal context buffer
        self.obs_history = deque(maxlen=self.temporal_context)

        # Action chunk buffer for temporal ensembling
        self.action_buffer = deque(maxlen=self.ensemble_window)

    def reset(self):
        """Reset temporal context and action buffers."""
        self.obs_history.clear()
        self.action_buffer.clear()

    def __call__(self, obs: BimanualObs) -> BimanualAction:
        """
        Predict action from observation using ACT policy.

        Args:
            obs: Current observation from simulator

        Returns:
            BimanualAction to execute
        """
        # Add current observation to history
        self.obs_history.append(obs)

        # If we don't have enough history yet, pad with current obs
        while len(self.obs_history) < self.temporal_context:
            self.obs_history.appendleft(obs)

        # Convert to tensor format
        tensor_obs = self._to_tensor_obs(list(self.obs_history))

        # Predict action chunk
        with torch.no_grad():
            action_chunk = self.model.predict_action_chunk(tensor_obs)  # [1, chunk_size, ACTION_SIZE]

        # Store action chunk for temporal ensembling
        self.action_buffer.append(action_chunk.cpu())

        # Temporal ensemble: average overlapping predictions
        if len(self.action_buffer) > 0:
            # For each buffer position i, we want action[i] from chunk[0]
            # This gives us multiple predictions for the current timestep
            current_actions = []
            for i, chunk in enumerate(self.action_buffer):
                step_in_chunk = len(self.action_buffer) - 1 - i
                if step_in_chunk < chunk.shape[1]:  # Make sure we're within chunk bounds
                    current_actions.append(chunk[0, step_in_chunk])

            if current_actions:
                # Average all predictions for current timestep
                action_array = torch.stack(current_actions).mean(dim=0).numpy()
            else:
                # Fallback: use first action from latest chunk
                action_array = action_chunk[0, 0].cpu().numpy()
        else:
            # No ensemble, just use first action from chunk
            action_array = action_chunk[0, 0].cpu().numpy()

        return BimanualAction(action_array)

    def _to_tensor_obs(self, obs_list):
        """Convert list of BimanualObs to TensorBimanualObs with temporal dimension."""
        # Stack observations: [temporal_context, num_cameras, H, W, 3]
        visual = np.stack([o.visual for o in obs_list], axis=0)  # [T, C, H, W, 3]
        visual = torch.from_numpy(visual).float().unsqueeze(0).to(self.device)  # [1, T, C, H, W, 3]

        # Stack qpos and qvel across temporal dimension: [1, T, joint_dim]
        qpos_arrays = [torch.from_numpy(o.qpos.array).float() for o in obs_list]
        qvel_arrays = [torch.from_numpy(o.qvel.array).float() for o in obs_list]
        qpos = torch.stack(qpos_arrays, dim=0).unsqueeze(0).to(self.device)  # [1, T, joint_dim]
        qvel = torch.stack(qvel_arrays, dim=0).unsqueeze(0).to(self.device)  # [1, T, joint_dim]

        # Wrap in TensorBimanualState - but we need to reshape for temporal context
        # The policy expects qpos/qvel to have .array attribute with shape [batch, temporal, ...]
        class TemporalTensorBimanualState:
            def __init__(self, tensor):
                self.array = tensor  # [batch, temporal, joint_dim]

        return TensorBimanualObs(
            visual=visual,
            qpos=TemporalTensorBimanualState(qpos),
            qvel=TemporalTensorBimanualState(qvel)
        )


# Create policy wrapper
act_policy = ACTPolicyWrapper(model, config, DEVICE)

# =============== CREATE SIMULATOR ==================
print("Creating MuJoCo simulator...")
sim = BimanualSim(
    merge_xml_files=[Path('robot/block.xml')],
    camera_dims=(config.image_size, config.image_size),
    obs_camera_names=config.camera_names,
    on_mujoco_init=randomize_block_position
)
print("✓ Simulator created!")
print()

# =============== ROLLOUT ==================
print(f"Starting rollout (max {MAX_STEPS} steps)...")
print("Note: First policy call may take 10-30 seconds due to CLIP encoding initialization.")
print()

# Rollout storage
left_frames, right_frames = [], []

# Use context manager for proper cleanup
with sim:
    # Reset policy for fresh rollout
    act_policy.reset()

    # Get initial observation
    obs = sim.get_obs()

    # Simulation rollout
    for step in tqdm(range(MAX_STEPS), desc="Rollout progress"):
        # Save images for video
        # Images from simulator are already in [0, 1] float range
        left_frames.append(obs.visual[0].copy())
        right_frames.append(obs.visual[1].copy())

        # Predict next action using ACT policy
        action = act_policy(obs)

        # Step simulation
        obs = sim.step(action)

print("✓ Rollout finished!")
print()

# =============== SAVE VIDEOS ==================
output_dir = Path("out/render-act")
output_dir.mkdir(parents=True, exist_ok=True)

left_path = output_dir / "left.mp4"
right_path = output_dir / "right.mp4"

print("Saving videos...")
save_frames_to_video(left_frames, left_path)
save_frames_to_video(right_frames, right_path)

print("✓ Videos saved:")
print(f"  Left camera:  {left_path}")
print(f"  Right camera:  {right_path}")
print()
