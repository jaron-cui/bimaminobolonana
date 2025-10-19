import torch
from encoders import build_encoder

def _pair(B=3, C=3, H=224, W=224):
    x = torch.randn(B, C, H, W)
    return (x, x.clone())

def _cfg(fuse):
    return {"name": "pri3d", "variant": "resnet18", "pretrained": False, "freeze": False, "out_dim": 512, "fuse": fuse}

def test_fusion_mean_max_shapes():
    for f in ["mean", "max"]:
        enc = build_encoder(_cfg(f))
        out = enc.encode(_pair())
        assert out["fused"].shape == (3, 512)

def test_fusion_concat_mlp_gated_bilinear_shapes_and_grad():
    for f in ["concat_mlp", "gated", "bilinear"]:
        enc = build_encoder(_cfg(f))
        # turn on train mode to allow grads through fusion heads
        enc.train()
        left, right = _pair()
        left.requires_grad_(True)
        right.requires_grad_(True)
        # use forward_single directly to avoid no_grad in encode()
        fl = enc.forward_single(left)
        fr = enc.forward_single(right)
        fused = enc._fuse(fl, fr)
        loss = fused.sum()
        loss.backward()
        # ensure gradients flowed to inputs (sanity)
        assert left.grad is not None and right.grad is not None
        # shape check
        assert fused.shape == (3, 512)
