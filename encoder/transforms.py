from __future__ import annotations
from typing import List, Optional, Tuple, Union
from PIL import Image
import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _resize_short_side(size: int) -> T.Resize:
    return T.Resize(size, interpolation=T.InterpolationMode.BICUBIC)

def build_image_transform(kind: str = "clip", size: int = 224, center_crop: bool = True) -> T.Compose:
    kind = kind.lower()
    if kind not in {"clip", "imagenet"}:
        raise ValueError(f"Unknown transform kind: {kind}")
    mean, std = (CLIP_MEAN, CLIP_STD) if kind == "clip" else (IMAGENET_MEAN, IMAGENET_STD)

    ops: List = [
        T.Lambda(lambda img: img if isinstance(img, Image.Image) else F.to_pil_image(img)),
        T.Lambda(lambda img: img.convert("RGB")),
        _resize_short_side(size),
    ]
    if center_crop:
        ops.append(T.CenterCrop(size))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)

def prepare_batch(
    images: Union[Tensor, "PIL.Image.Image", List[Union["PIL.Image.Image", Tensor]]],
    transform: Optional[callable] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    def _to_tensor(img) -> Tensor:
        if transform is not None:
            return transform(img)
        if isinstance(img, torch.Tensor) and img.ndim == 3 and img.shape[0] == 3:
            return img.to(dtype)
        raise ValueError("Provide a transform for non-tensor inputs or use 3xHxW tensors.")

    if isinstance(images, torch.Tensor):
        batch = images if images.ndim == 4 else images.unsqueeze(0)
    elif isinstance(images, list):
        batch = torch.stack([_to_tensor(im) for im in images], dim=0)
    else:
        batch = _to_tensor(images).unsqueeze(0)

    if device is not None:
        batch = batch.to(device)
    if batch.dtype != dtype:
        batch = batch.to(dtype)
    return batch
