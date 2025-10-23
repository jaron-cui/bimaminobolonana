import torch
from encoder.transforms import build_image_transform, prepare_batch

def _dummy_image(h=240, w=320):
    # uint8 HxWxC array; ToPILImage handles numpy input
    x = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8)
    return x.numpy()

def test_clip_transform_and_batch():
    tfm = build_image_transform(kind="clip", size=224)
    img = _dummy_image()
    batch = prepare_batch(img, transform=tfm)
    assert batch.shape == (1, 3, 224, 224)
    assert batch.dtype == torch.float32
    assert torch.isfinite(batch).all()

def test_imagenet_transform_and_list_batch():
    tfm = build_image_transform(kind="imagenet", size=256)
    imgs = [_dummy_image(), _dummy_image()]
    batch = prepare_batch(imgs, transform=tfm)
    assert batch.shape == (2, 3, 256, 256)

def test_prepare_batch_tensor_direct():
    x = torch.randn(3, 224, 224)
    batch = prepare_batch(x)
    assert batch.shape == (1, 3, 224, 224)
