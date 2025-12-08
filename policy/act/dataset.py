"""
Temporal dataset wrapper for ACT training.

Wraps the BimanualDataset to provide temporal context (sequences of observations)
and action chunks (sequences of future actions) needed for ACT training.
"""

from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from train.dataset import BimanualDataset, TensorBimanualObs, TensorBimanualAction, TensorBimanualState, BimanualDatasetMetadata, JOINT_OBSERVATION_SIZE, ACTION_SIZE


class TemporalBimanualDataset(Dataset):
    """
    Dataset wrapper that provides temporal context for ACT training.

    Returns:
    - obs_sequence: Past temporal_context observations [temporal_context, ...]
    - action_chunk: Future chunk_size actions [chunk_size, ACTION_SIZE]
    """

    def __init__(
            self,
            base_dataset: BimanualDataset,
            temporal_context: int = 3,
            chunk_size: int = 50,
            pad_mode: str = 'repeat',  # 'repeat' or 'zero'
            action_mean_path: str | None = None,
            action_std_path: str | None = None,
            action_mean: torch.Tensor | None = None,
            action_std: torch.Tensor | None = None,
            use_relative: bool = True,
        ):
        """
        Initialize temporal dataset wrapper.

        Args:
            base_dataset: Base BimanualDataset to wrap
            temporal_context: Number of past observations to include
            chunk_size: Number of future actions to predict
            pad_mode: How to pad at episode boundaries ('repeat' or 'zero')
            action_mean: Optional pre-loaded mean tensor
            action_std: Optional pre-loaded std tensor
        """
        self.base_dataset = base_dataset
        self.temporal_context = temporal_context
        self.chunk_size = chunk_size
        self.pad_mode = pad_mode
        # Normalization / relative action settings
        self.use_relative = use_relative
        self.action_mean = None
        self.action_std = None

        # Priority: 1. Passed tensors, 2. Paths
        if action_mean is not None and action_std is not None:
             self.action_mean = action_mean.float()
             self.action_std = action_std.float()
        elif action_mean_path is not None and action_std_path is not None:
            import numpy as _np
            try:
                mean = _np.load(action_mean_path)
                std = _np.load(action_std_path)
                # convert to torch tensors (1D)
                import torch as _torch
                self.action_mean = _torch.from_numpy(mean).float()
                self.action_std = _torch.from_numpy(std).float()
            except Exception:
                # If loading fails, leave as None and proceed without normalization
                self.action_mean = None
                self.action_std = None

        # Build episode boundaries
        self.episode_boundaries = self._build_episode_boundaries()

    def _build_episode_boundaries(self) -> List[Tuple[int, int]]:
        """
        Build list of (start_idx, end_idx) for each episode.

        Returns:
            List of (start, end) tuples for each episode
        """
        metadata = self.base_dataset.metadata
        boundaries = []
        current_idx = 0

        for rollout_length in metadata.rollout_lengths:
            boundaries.append((current_idx, current_idx + rollout_length))
            current_idx += rollout_length

        return boundaries

    def _find_episode(self, idx: int) -> Tuple[int, int, int]:
        """
        Find which episode contains the given index.

        Args:
            idx: Global dataset index

        Returns:
            (episode_idx, start_idx, end_idx)
        """
        for episode_idx, (start, end) in enumerate(self.episode_boundaries):
            if start <= idx < end:
                return episode_idx, start, end
        raise IndexError(f"Index {idx} not found in any episode")

    def __len__(self) -> int:
        """
        Length is the same as base dataset, but we filter out samples
        where we can't get a full action chunk.
        """
        # We can only use samples where we have enough future actions
        valid_count = 0
        for start, end in self.episode_boundaries:
            episode_length = end - start
            # Each sample needs chunk_size future actions
            valid_count += max(0, episode_length - self.chunk_size + 1)
        return valid_count

    def _map_to_base_idx(self, temporal_idx: int) -> int:
        """
        Map temporal dataset index to base dataset index.

        Args:
            temporal_idx: Index in temporal dataset

        Returns:
            Corresponding index in base dataset
        """
        count = 0
        for start, end in self.episode_boundaries:
            episode_valid = end - start - self.chunk_size + 1
            if episode_valid <= 0:
                continue
            if count + episode_valid > temporal_idx:
                # Found the episode
                offset = temporal_idx - count
                return start + offset
            count += episode_valid
        raise IndexError(f"Temporal index {temporal_idx} out of range")

    def __getitem__(self, idx: int) -> Tuple[TensorBimanualObs, TensorBimanualAction]:
        """
        Get a temporal observation sequence and action chunk.

        Args:
            idx: Index in the temporal dataset

        Returns:
            (obs_sequence, action_chunk) where:
            - obs_sequence: TensorBimanualObs with temporal dimension
                visual: [temporal_context, 2, H, W, 3]
                qpos: [temporal_context, JOINT_OBS_SIZE]
                qvel: [temporal_context, JOINT_OBS_SIZE]
            - action_chunk: TensorBimanualAction [chunk_size, ACTION_SIZE]
        """
        # Map to base dataset index
        base_idx = self._map_to_base_idx(idx)
        episode_idx, episode_start, episode_end = self._find_episode(base_idx)

        # Collect temporal context observations
        obs_list = []
        for t in range(self.temporal_context):
            context_idx = base_idx - (self.temporal_context - 1 - t)

            # Handle padding at episode boundaries
            if context_idx < episode_start:
                if self.pad_mode == 'repeat':
                    # Repeat first observation
                    context_idx = episode_start
                else:
                    # Use zero padding (handled below)
                    context_idx = episode_start

            obs, _ = self.base_dataset[context_idx]
            obs_list.append(obs)

        # Stack temporal observations
        visual_seq = torch.stack([obs.visual for obs in obs_list], dim=1)  # [1, temporal_context, 2, H, W, 3]
        qpos_seq = torch.stack([obs.qpos.array for obs in obs_list], dim=1)  # [1, temporal_context, JOINT_OBS_SIZE]
        qvel_seq = torch.stack([obs.qvel.array for obs in obs_list], dim=1)  # [1, temporal_context, JOINT_OBS_SIZE]

        # Remove batch dimension (we'll add it back in collate)
        visual_seq = visual_seq.squeeze(0)  # [temporal_context, 2, H, W, 3]
        qpos_seq = qpos_seq.squeeze(0)  # [temporal_context, JOINT_OBS_SIZE]
        qvel_seq = qvel_seq.squeeze(0)  # [temporal_context, JOINT_OBS_SIZE]

        obs_sequence = TensorBimanualObs(
            visual=visual_seq,
            qpos=TensorBimanualState(qpos_seq),
            qvel=TensorBimanualState(qvel_seq)
        )

        # Collect action chunk
        action_list = []
        for t in range(self.chunk_size):
            action_idx = base_idx + t

            # Make sure we don't go beyond episode boundary
            if action_idx >= episode_end:
                # Repeat last action if needed
                action_idx = episode_end - 1

            _, action = self.base_dataset[action_idx]
            action_list.append(action.array)

        # Stack actions into chunk
        action_chunk = torch.stack(action_list, dim=1)  # [1, chunk_size, ACTION_SIZE]
        action_chunk = action_chunk.squeeze(0)  # [chunk_size, ACTION_SIZE]

        # Convert to relative actions (optional) and normalize (optional)
        # Compute approximate action from last context qpos
        # qpos_seq: [temporal_context, JOINT_OBS_SIZE]
        try:
            # Create a TensorBimanualState for the last timestep and get its approx action
            last_qpos = qpos_seq[-1].unsqueeze(0)  # [1, JOINT_OBS_SIZE]
            approx_action = TensorBimanualState(last_qpos).to_approximate_action().array.squeeze(0)
        except Exception:
            approx_action = None

        rel = action_chunk
        if self.use_relative and approx_action is not None:
            # subtract approx from all timesteps in chunk
            rel = action_chunk - approx_action.unsqueeze(0)

        normed = rel
        if self.action_mean is not None and self.action_std is not None:
            # broadcast mean/std over chunk dimension
            mean = self.action_mean.unsqueeze(0)
            std = self.action_std.unsqueeze(0)
            # avoid division by zero
            std = torch.where(std == 0, torch.ones_like(std), std)
            normed = (rel - mean) / std

        # Wrap in TensorBimanualAction with chunk dimension
        action_chunk_wrapped = TensorBimanualAction(normed)

        return obs_sequence, action_chunk_wrapped

    @staticmethod
    def collate_fn(
        batch: List[Tuple[TensorBimanualObs, TensorBimanualAction]]
    ) -> Tuple[TensorBimanualObs, TensorBimanualAction]:
        """
        Collate batch of temporal observations and action chunks.

        Args:
            batch: List of (obs_sequence, action_chunk) tuples

        Returns:
            Batched (obs_sequence, action_chunk)
        """
        obs_sequences, action_chunks = zip(*batch)

        # Stack temporal observations
        # visual: [temporal_context, 2, H, W, 3] -> [batch, temporal_context, 2, H, W, 3]
        visual_batch = torch.stack([obs.visual for obs in obs_sequences], dim=0)
        qpos_batch = torch.stack([obs.qpos.array for obs in obs_sequences], dim=0)
        qvel_batch = torch.stack([obs.qvel.array for obs in obs_sequences], dim=0)

        # Create a simple container for temporal states (not TensorBimanualState which expects 2D)
        class TemporalState:
            def __init__(self, array):
                self.array = array

            def to(self, *args, **kwargs):
                return TemporalState(self.array.to(*args, **kwargs))

            @property
            def device(self):
                return self.array.device

        batched_obs = TensorBimanualObs(
            visual=visual_batch,
            qpos=TemporalState(qpos_batch),  # [batch, temporal, dim]
            qvel=TemporalState(qvel_batch)   # [batch, temporal, dim]
        )

        # Stack action chunks: [chunk_size, ACTION_SIZE] -> [batch, chunk_size, ACTION_SIZE]
        action_batch = torch.stack([action.array for action in action_chunks], dim=0)

        # Similarly, create container for temporal actions
        class TemporalAction:
            def __init__(self, array):
                self.array = array

            def to(self, *args, **kwargs):
                return TemporalAction(self.array.to(*args, **kwargs))

            @property
            def device(self):
                return self.array.device

        batched_actions = TemporalAction(action_batch)  # [batch, chunk_size, ACTION_SIZE]

        return batched_obs, batched_actions


def create_temporal_dataloader(
    dataset_path: str,
    temporal_context: int = 3,
    chunk_size: int = 50,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    action_mean_path: str | None = None,
    action_std_path: str | None = None,
) -> torch.utils.data.DataLoader:
    """
    Create a temporal dataloader for ACT training.

    Args:
        dataset_path: Path to BimanualDataset
        temporal_context: Number of past observations
        chunk_size: Number of future actions
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers

    Returns:
        DataLoader for temporal dataset
    """
    base_dataset = BimanualDataset(dataset_path)
    temporal_dataset = TemporalBimanualDataset(
        base_dataset=base_dataset,
        temporal_context=temporal_context,
        chunk_size=chunk_size,
        action_mean_path=action_mean_path,
        action_std_path=action_std_path,
    )

    dataloader = torch.utils.data.DataLoader(
        temporal_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=TemporalBimanualDataset.collate_fn,
    )

    return dataloader
