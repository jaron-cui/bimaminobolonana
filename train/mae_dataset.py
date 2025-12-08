"""
Dataset for MAE pretraining on bimanual manipulation data.

Loads image pairs (left, right cameras) from BimanualDataset format
and prepares them for CLIP-based MAE training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .dataset import BimanualDataset, BimanualDatasetMetadata


# CLIP normalization stats
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


class MAEBimanualDataset(Dataset):
    """
    Dataset for MAE pretraining on bimanual image pairs.

    Loads images from BimanualDataset format and prepares them for CLIP:
    - Extracts left and right camera images
    - Resizes from source resolution (e.g., 128x128) to 224x224
    - Normalizes with CLIP statistics

    Args:
        data_directory: Path to dataset directory containing observations.npy
        img_size: Target image size for CLIP (default 224)
        augment: Apply data augmentation (random horizontal flip, color jitter)
    """

    def __init__(
        self,
        data_directory: Union[Path, str],
        img_size: int = 224,
        augment: bool = True,
    ):
        super().__init__()

        self.data_directory = Path(data_directory)
        self.img_size = img_size
        self.augment = augment

        # Load metadata
        metadata = BimanualDatasetMetadata.from_file(data_directory, read_only=True)
        if metadata is None:
            raise FileNotFoundError(f'Dataset not found in {data_directory}.')

        self.metadata = metadata
        self.camera_height = metadata.camera_height
        self.camera_width = metadata.camera_width
        self.sample_count = metadata.sample_count

        # Calculate visual size in observation array
        self.visual_shape = (2, self.camera_height, self.camera_width, 3)
        self.visual_size = np.prod(self.visual_shape)

        # Load observation array (memory-mapped for efficiency)
        memmap = metadata.memmap_data(overwrite=False)
        if memmap is None:
            raise RuntimeError(f'Failed to load dataset from {data_directory}')

        observation_array, _ = memmap
        self._observation_array = observation_array

        # Register normalization stats as buffers
        self.register_buffer_mean = CLIP_MEAN
        self.register_buffer_std = CLIP_STD

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.

        Returns:
            dict with:
                - left_img: (3, 224, 224) left camera image, normalized
                - right_img: (3, 224, 224) right camera image, normalized
                - left_raw: (3, H, W) raw left image before normalization (for reconstruction target)
                - right_raw: (3, H, W) raw right image before normalization
        """
        if index >= self.sample_count:
            raise IndexError(f'Index {index} out of range [0, {self.sample_count})')

        # Extract visual data
        obs = self._observation_array[index]
        visual = obs[:self.visual_size].reshape(self.visual_shape)  # (2, H, W, 3)

        # Convert to tensor and rearrange to (C, H, W)
        # Copy to avoid non-writable tensor warning
        left_img = torch.from_numpy(visual[0].copy()).permute(2, 0, 1).float()   # (3, H, W)
        right_img = torch.from_numpy(visual[1].copy()).permute(2, 0, 1).float()  # (3, H, W)

        # Images are in [0, 1] range (from dataset generation)
        # Clamp to ensure valid range
        left_img = left_img.clamp(0, 1)
        right_img = right_img.clamp(0, 1)

        # Apply augmentation (same to both views)
        if self.augment and self.training:
            left_img, right_img = self._augment(left_img, right_img)

        # Resize to target size
        left_resized = self._resize(left_img)
        right_resized = self._resize(right_img)

        # Normalize with CLIP stats
        left_norm = self._normalize(left_resized)
        right_norm = self._normalize(right_resized)

        return {
            "left_img": left_norm,
            "right_img": right_norm,
            "left_raw": left_resized,  # For reconstruction loss
            "right_raw": right_resized,
        }

    def _resize(self, img: torch.Tensor) -> torch.Tensor:
        """Resize image to target size."""
        if img.shape[-2:] == (self.img_size, self.img_size):
            return img

        return F.interpolate(
            img.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize with CLIP statistics."""
        mean = CLIP_MEAN.view(3, 1, 1).to(img.device)
        std = CLIP_STD.view(3, 1, 1).to(img.device)
        return (img - mean) / std

    def _augment(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply consistent augmentation to both views.

        Same random parameters for left and right to maintain correspondence.
        """
        # Random horizontal flip (same for both)
        if torch.rand(1).item() > 0.5:
            left = torch.flip(left, dims=[-1])
            right = torch.flip(right, dims=[-1])

        # Random color jitter (same parameters for both)
        if torch.rand(1).item() > 0.5:
            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.2  # [0.9, 1.1]
            left = (left * brightness).clamp(0, 1)
            right = (right * brightness).clamp(0, 1)

        return left, right

    @property
    def training(self) -> bool:
        """Check if in training mode (for augmentation)."""
        return getattr(self, '_training', True)

    def train(self, mode: bool = True):
        """Set training mode."""
        self._training = mode
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)


class MAEMultiDataset(Dataset):
    """
    Combine multiple MAE datasets.

    Useful for training on multiple data directories together.
    """

    def __init__(
        self,
        data_directories: List[Union[Path, str]],
        img_size: int = 224,
        augment: bool = True,
    ):
        super().__init__()

        self.datasets = [
            MAEBimanualDataset(d, img_size=img_size, augment=augment)
            for d in data_directories
        ]

        # Compute cumulative lengths
        self.lengths = [len(d) for d in self.datasets]
        self.cumulative = np.cumsum([0] + self.lengths)
        self.total_length = sum(self.lengths)

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative[1:], index, side='right')
        local_idx = index - self.cumulative[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

    def train(self, mode: bool = True):
        for d in self.datasets:
            d.train(mode)
        return self

    def eval(self):
        return self.train(False)


def create_mae_dataloader(
    data_directory: Union[Path, str, List[Union[Path, str]]],
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
    augment: bool = True,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for MAE pretraining.

    Args:
        data_directory: Path(s) to dataset directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Target image size
        augment: Apply augmentation
        shuffle: Shuffle data
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    if isinstance(data_directory, (list, tuple)):
        dataset = MAEMultiDataset(
            data_directories=data_directory,
            img_size=img_size,
            augment=augment,
        )
    else:
        dataset = MAEBimanualDataset(
            data_directory=data_directory,
            img_size=img_size,
            augment=augment,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for consistent batch norm
    )


def denormalize_clip(img: torch.Tensor) -> torch.Tensor:
    """
    Denormalize CLIP-normalized images for visualization.

    Args:
        img: (B, 3, H, W) or (3, H, W) normalized image

    Returns:
        Denormalized image in [0, 1] range
    """
    mean = CLIP_MEAN.view(1, 3, 1, 1) if img.dim() == 4 else CLIP_MEAN.view(3, 1, 1)
    std = CLIP_STD.view(1, 3, 1, 1) if img.dim() == 4 else CLIP_STD.view(3, 1, 1)

    mean = mean.to(img.device)
    std = std.to(img.device)

    return (img * std + mean).clamp(0, 1)
