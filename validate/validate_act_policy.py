"""
Evaluate ACT (Action Chunking Transformer) Policy Success Rate

This script loads a trained ACT checkpoint and evaluates its success rate
on the bimanual manipulation task using simulation rollouts.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from collections import deque

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from policy.act import build_act_policy
from robot.sim import BimanualAction, BimanualObs, BimanualSim, randomize_block_position
from validate.evaluation import evaluate_policy
from train.dataset import TensorBimanualObs

# ------------------------
# CONFIGURATION
# ------------------------

# Path to your checkpoint
CHECKPOINT_PATH = "runs/act_policy/150.pt"

# Path to config (use default config if not saved with checkpoint)
CONFIG_PATH = "configs/policy_act.yaml"

# Evaluation parameters
NUM_ROLLOUTS = 5
MAX_STEPS_PER_ROLLOUT = 400
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------
# LOAD MODEL
# ------------------------

print(f"Loading config from: {CONFIG_PATH}")
config = OmegaConf.load(CONFIG_PATH)
print(f"Config loaded: {config.name}")
print(f"  - Chunk size: {config.chunk_size}")
print(f"  - Temporal context: {config.temporal_context}")
print(f"  - Encoder: {config.encoder.name}")
print(f"  - Image size: {config.image_size}")
print()

print(f"Building ACT policy...")
model = build_act_policy(config).to(DEVICE)
model.eval()

print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
print("✓ Checkpoint loaded successfully!")
print()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters:")
print(f"  - Total: {total_params:,}")
print(f"  - Trainable: {trainable_params:,}")
print()

# ------------------------
# POLICY WRAPPER
# ------------------------

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
        from train.dataset import TensorBimanualState

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
        # So we create a wrapper that provides .array
        class TemporalTensorBimanualState:
            def __init__(self, tensor):
                self.array = tensor  # [batch, temporal, joint_dim]

        return TensorBimanualObs(
            visual=visual,
            qpos=TemporalTensorBimanualState(qpos),
            qvel=TemporalTensorBimanualState(qvel)
        )


# Create policy wrapper
act_policy_wrapper = ACTPolicyWrapper(model, config, DEVICE)

# ------------------------
# DEFINE POLICY AND SIM CREATOR
# ------------------------

def act_policy(obs: BimanualObs) -> BimanualAction:
    """Policy function that wraps the ACT model."""
    return act_policy_wrapper(obs)


def create_sim() -> BimanualSim:
    """Create a fresh simulation environment for each rollout."""
    # Reset the policy wrapper for each new rollout
    act_policy_wrapper.reset()

    # Create simulator with same camera settings as training
    print("  Creating simulation...", end="", flush=True)
    sim = BimanualSim(
        merge_xml_files=[Path('robot/block.xml')],
        on_mujoco_init=randomize_block_position,
        camera_dims=(config.image_size, config.image_size),
        obs_camera_names=config.camera_names
    )
    print(" done.", flush=True)
    return sim


# ------------------------
# CUSTOM EVALUATION WITH DEBUG OUTPUT
# ------------------------

def run_evaluation_rollout_debug(
    policy, sim, max_steps_per_rollout, verbose=True
):
    """Custom rollout function with debug output."""
    from validate.evaluation import GripperTracker
    import mujoco

    left_gripper_tracker = GripperTracker('left')
    right_gripper_tracker = GripperTracker('right')

    with sim as sim:
        if verbose:
            print("    Getting initial observation...", end="", flush=True)
        obs = sim.get_obs()
        if verbose:
            print(" done.", flush=True)

        for step in range(max_steps_per_rollout):
            if verbose and step == 0:
                print("    Running first policy call (CLIP encoding may take 10-30s)...", end="", flush=True)
            elif verbose and step % 50 == 0:
                print(f"    Step {step}/{max_steps_per_rollout}...", end="", flush=True)

            action = policy(obs)

            if verbose and step == 0:
                print(" done.", flush=True)
            elif verbose and step % 50 == 0:
                print(" done.", flush=True)

            obs = sim.step(action)

            left_gripper_tracker.update(action, obs)
            right_gripper_tracker.update(action, obs)

            block_pos = sim.data.xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'block')]
            right_gripper_distance = np.linalg.norm(block_pos - right_gripper_tracker.pos()).item()
            left_gripper_distance = np.linalg.norm(block_pos - left_gripper_tracker.pos()).item()
            if right_gripper_tracker.is_gripping() and right_gripper_distance < 0.05 and left_gripper_distance > 0.2:
                if verbose:
                    print(f"    ✓ Success at step {step}!", flush=True)
                return True

    if verbose:
        print(f"    ✗ Failed (max steps reached)", flush=True)
    return False


# ------------------------
# RUN EVALUATION
# ------------------------

print("=" * 60)
print("Starting ACT Policy Evaluation")
print("=" * 60)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Number of rollouts: {NUM_ROLLOUTS}")
print(f"Max steps per rollout: {MAX_STEPS_PER_ROLLOUT}")
print(f"Temporal context: {config.temporal_context} frames")
print(f"Action chunk size: {config.chunk_size}")
print(f"Temporal ensemble: {config.get('temporal_ensemble', False)}")
if config.get('temporal_ensemble', False):
    print(f"Ensemble window: {config.get('ensemble_window', 10)}")
print("=" * 60)
print()

success_count = 0
for i in range(NUM_ROLLOUTS):
    print(f"\nRollout {i+1}/{NUM_ROLLOUTS}:", flush=True)
    if run_evaluation_rollout_debug(act_policy, create_sim(), MAX_STEPS_PER_ROLLOUT, verbose=True):
        success_count += 1

success_rate = success_count / NUM_ROLLOUTS

print()
print("=" * 60)
print(f"ACT Policy Success Rate: {success_rate * 100:.2f}%")
print(f"Successful rollouts: {int(success_rate * NUM_ROLLOUTS)}/{NUM_ROLLOUTS}")
print("=" * 60)
