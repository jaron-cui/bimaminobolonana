"""
MAE components for CLIP-based bimanual pretraining.

Inspired by "Touch in the Wild" cross-modal attention approach.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class CLIPPatchExtractor(nn.Module):
    """
    Extract patch tokens from CLIP ViT encoder.

    CLIP ViT structure:
    1. conv1 (patch embedding): img -> patch tokens
    2. positional embedding added
    3. class token prepended
    4. transformer blocks
    5. final projection (we skip this for MAE)

    We hook into the model to get intermediate patch tokens.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        freeze: bool = False,
    ):
        super().__init__()

        # Load CLIP model
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = model.visual

        # Get model dimensions
        self.embed_dim = self.visual.output_dim  # Final output dim (512 for ViT-B-32)
        self.hidden_dim = self.visual.transformer.width  # Hidden dim (768 for ViT-B)
        self.patch_size = self.visual.conv1.kernel_size[0]  # 32 for ViT-B-32
        self.num_heads = self.visual.transformer.resblocks[0].attn.num_heads  # 12 for ViT-B

        # Freeze if requested
        if freeze:
            for p in self.visual.parameters():
                p.requires_grad = False

    def get_patch_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patch tokens from image.

        Args:
            x: (B, 3, H, W) input images, should be 224x224 for CLIP

        Returns:
            patch_tokens: (B, N, D) patch tokens (excluding CLS token)
            cls_token: (B, D) CLS token
        """
        # Patch embedding
        x = self.visual.conv1(x)  # (B, D, H/P, W/P)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, D, N)
        x = x.permute(0, 2, 1)  # (B, N, D)

        # Add class token
        cls_token = self.visual.class_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        cls_token = cls_token.expand(x.shape[0], -1, -1)  # (B, 1, D)
        x = torch.cat([cls_token, x], dim=1)  # (B, N+1, D)

        # Add positional embedding
        x = x + self.visual.positional_embedding

        # Pre-transformer LayerNorm
        x = self.visual.ln_pre(x)

        # Run through transformer
        x = x.permute(1, 0, 2)  # (N+1, B, D) for transformer
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # (B, N+1, D)

        # Split CLS token and patch tokens
        cls_token = x[:, 0]  # (B, D)
        patch_tokens = x[:, 1:]  # (B, N, D)

        return patch_tokens, cls_token

    def get_patch_tokens_with_mask(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract patch tokens with random masking.

        Args:
            x: (B, 3, H, W) input images
            mask_ratio: fraction of patches to mask

        Returns:
            visible_tokens: (B, N_vis, D) visible patch tokens
            ids_restore: (B, N) indices to restore original order
            mask: (B, N) binary mask (1 = masked, 0 = visible)
        """
        # Patch embedding
        x = self.visual.conv1(x)  # (B, D, H/P, W/P)
        B, D, H, W = x.shape
        N = H * W  # Total number of patches

        x = x.reshape(B, D, N).permute(0, 2, 1)  # (B, N, D)

        # Random masking
        num_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep only visible tokens
        ids_keep = ids_shuffle[:, :num_keep]
        visible_tokens = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # Create mask (1 = masked, 0 = visible)
        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_keep, 0)

        # Add class token
        cls_token = self.visual.class_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        cls_token = cls_token.expand(B, -1, -1)  # (B, 1, D)
        visible_tokens = torch.cat([cls_token, visible_tokens], dim=1)  # (B, 1+N_vis, D)

        # Add positional embedding (only for visible positions)
        cls_pos = self.visual.positional_embedding[0:1]  # (1, D)
        vis_pos = torch.gather(
            self.visual.positional_embedding[1:].unsqueeze(0).expand(B, -1, -1),
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        pos_embed = torch.cat([cls_pos.unsqueeze(0).expand(B, -1, -1), vis_pos], dim=1)
        visible_tokens = visible_tokens + pos_embed

        # Pre-transformer LayerNorm
        visible_tokens = self.visual.ln_pre(visible_tokens)

        # Run through transformer
        visible_tokens = visible_tokens.permute(1, 0, 2)  # (N_vis+1, B, D)
        visible_tokens = self.visual.transformer(visible_tokens)
        visible_tokens = visible_tokens.permute(1, 0, 2)  # (B, N_vis+1, D)

        # Remove CLS token for decoder
        visible_tokens = visible_tokens[:, 1:]  # (B, N_vis, D)

        return visible_tokens, ids_restore, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward: return pooled CLS token features.
        """
        _, cls_token = self.get_patch_tokens(x)
        # Apply final LayerNorm and projection
        cls_token = self.visual.ln_post(cls_token)
        if self.visual.proj is not None:
            cls_token = cls_token @ self.visual.proj
        return cls_token


class CrossAttentionBlock(nn.Module):
    """
    Cross attention block for bimanual view fusion.
    Query attends to key-value from another view.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, D) query tokens
            key_value: (B, N_kv, D) key-value tokens from other view

        Returns:
            (B, N_q, D) updated query tokens
        """
        # Cross attention
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x = query + attn_out

        # FFN
        x = x + self.ffn(self.norm_ffn(x))

        return x


class BimanualCrossAttention(nn.Module):
    """
    Bidirectional cross attention between left and right view tokens.

    Inspired by "Touch in the Wild" cross-modal attention for
    learning correspondences between modalities.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        left_tokens: torch.Tensor,
        right_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross attention between left and right tokens.

        Args:
            left_tokens: (B, N_left, D) left view tokens
            right_tokens: (B, N_right, D) right view tokens

        Returns:
            fused_left: (B, N_left, D) left tokens attended by right
            fused_right: (B, N_right, D) right tokens attended by left
        """
        for layer in self.layers:
            # Left attends to Right
            left_new = layer(query=left_tokens, key_value=right_tokens)
            # Right attends to Left
            right_new = layer(query=right_tokens, key_value=left_tokens)

            left_tokens = left_new
            right_tokens = right_new

        return left_tokens, right_tokens


class MAEDecoder(nn.Module):
    """
    Lightweight decoder for MAE reconstruction.

    Takes encoded visible tokens + mask tokens and reconstructs image patches.
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        decoder_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_patches: int = 49,  # 7x7 for 224x224 with patch_size=32
        patch_size: int = 32,
        in_channels: int = 3,
    ):
        super().__init__()

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Project encoder dim to decoder dim
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional embedding for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_dim)
        )
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=num_heads,
                dim_feedforward=int(decoder_dim * mlp_ratio),
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Predict pixel values for each patch
        self.decoder_pred = nn.Linear(
            decoder_dim,
            patch_size * patch_size * in_channels
        )

    def forward(
        self,
        visible_tokens: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode visible tokens to reconstruct full image.

        Args:
            visible_tokens: (B, N_vis, D_enc) encoded visible tokens
            ids_restore: (B, N) indices to restore original order

        Returns:
            pred: (B, N, patch_size^2 * 3) predicted pixel values per patch
        """
        B, N_vis, D_enc = visible_tokens.shape
        N = ids_restore.shape[1]  # Total number of patches

        # Project to decoder dimension
        x = self.decoder_embed(visible_tokens)  # (B, N_vis, D_dec)
        D_dec = x.shape[-1]

        # Append mask tokens
        mask_tokens = self.mask_token.expand(B, N - N_vis, -1)

        # Create full sequence with mask tokens in place
        x_full = torch.zeros(B, N, D_dec, device=x.device, dtype=x.dtype)

        # Scatter visible tokens back to original positions
        vis_indices = torch.argsort(ids_restore, dim=1)[:, :N_vis]
        x_full.scatter_(
            1,
            vis_indices.unsqueeze(-1).expand(-1, -1, D_dec),
            x
        )

        # Fill mask positions
        mask_indices = torch.argsort(ids_restore, dim=1)[:, N_vis:]
        x_full.scatter_(
            1,
            mask_indices.unsqueeze(-1).expand(-1, -1, D_dec),
            mask_tokens
        )

        # Add positional embedding
        x_full = x_full + self.decoder_pos_embed

        # Apply decoder blocks
        for block in self.decoder_blocks:
            x_full = block(x_full)

        x_full = self.decoder_norm(x_full)

        # Predict pixel values
        pred = self.decoder_pred(x_full)  # (B, N, patch_size^2 * 3)

        return pred

    def unpatchify(self, pred: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Convert patch predictions back to image.

        Args:
            pred: (B, N, patch_size^2 * 3) predicted patches
            h, w: number of patches in height and width

        Returns:
            img: (B, 3, H, W) reconstructed image
        """
        B = pred.shape[0]
        p = self.patch_size
        c = self.in_channels

        pred = pred.reshape(B, h, w, p, p, c)
        pred = pred.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        pred = pred.reshape(B, c, h * p, w * p)

        return pred


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert images to patches for loss computation.

    Args:
        imgs: (B, 3, H, W)
        patch_size: size of each patch

    Returns:
        patches: (B, N, patch_size^2 * 3)
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0

    h = H // patch_size
    w = W // patch_size

    imgs = imgs.reshape(B, C, h, patch_size, w, patch_size)
    imgs = imgs.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
    patches = imgs.reshape(B, h * w, patch_size * patch_size * C)

    return patches


def compute_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    norm_pix_loss: bool = True,
    use_pixel_mask: bool = False,
    pixel_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute MAE loss on masked patches/pixels only.

    Args:
        pred: (B, N, D) predicted patch values
        target: (B, N, D) target patch values
        mask: (B, N) binary mask (1 = masked patch, 0 = visible patch)
        norm_pix_loss: normalize target by patch variance
        use_pixel_mask: if True, use pixel-level masking
        pixel_mask: (B, N, D) pixel-level mask, required if use_pixel_mask=True

    Returns:
        loss: scalar loss value
    """
    if norm_pix_loss:
        # Normalize target by patch mean and variance
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

    # MSE loss per pixel
    loss = (pred - target) ** 2  # (B, N, D)

    if use_pixel_mask and pixel_mask is not None:
        # Pixel-level masking: compute loss only on masked pixels
        masked_loss = loss * pixel_mask
        total_masked_pixels = pixel_mask.sum()

        if total_masked_pixels > 0:
            loss = masked_loss.sum() / total_masked_pixels
        else:
            loss = masked_loss.sum()  # Fallback
    else:
        # Original: patch-level masking only
        loss = loss.mean(dim=-1)  # (B, N) mean over patch pixels
        loss = (loss * mask).sum() / mask.sum()

    return loss


def generate_sparse_pixel_mask(
    batch_size: int,
    num_patches: int,
    pixels_per_patch: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a sparse pixel-level mask across all patches.

    Unlike patch-level masking (large 32x32 blocks), this masks individual pixels
    randomly across the entire image, creating a fine-grained scattered pattern.

    Args:
        batch_size: B
        num_patches: N (number of patches, e.g., 49 for 7x7)
        pixels_per_patch: D (pixels per patch, e.g., 32*32*3 = 3072)
        mask_ratio: fraction of total pixels to mask (e.g., 0.75)
        device: torch device

    Returns:
        pixel_mask: (B, N, D) binary mask (1 = masked pixel, 0 = visible)
    """
    total_pixels = num_patches * pixels_per_patch
    num_mask = int(total_pixels * mask_ratio)

    # Create mask for each sample in batch
    pixel_mask = torch.zeros(batch_size, total_pixels, device=device)

    for b in range(batch_size):
        # Randomly select pixels to mask
        noise = torch.rand(total_pixels, device=device)
        ids_shuffle = torch.argsort(noise)
        ids_mask = ids_shuffle[:num_mask]
        pixel_mask[b, ids_mask] = 1.0

    # Reshape to (B, N, D)
    pixel_mask = pixel_mask.reshape(batch_size, num_patches, pixels_per_patch)

    return pixel_mask


def pixel_mask_to_image(
    pixel_mask: torch.Tensor,
    patch_size: int,
    img_size: int,
) -> torch.Tensor:
    """
    Convert pixel mask from patch format to image format for visualization.

    Args:
        pixel_mask: (B, N, D) pixel-level mask where D = patch_size^2 * 3
        patch_size: size of each patch
        img_size: size of the image

    Returns:
        mask_img: (B, 3, H, W) mask in image format
    """
    B, N, D = pixel_mask.shape
    h = w = img_size // patch_size
    c = 3

    # D = patch_size * patch_size * 3
    # Reshape: (B, N, D) -> (B, h, w, p, p, c) -> (B, c, H, W)
    pixel_mask = pixel_mask.reshape(B, h, w, patch_size, patch_size, c)
    pixel_mask = pixel_mask.permute(0, 5, 1, 3, 2, 4)  # (B, c, h, p, w, p)
    mask_img = pixel_mask.reshape(B, c, h * patch_size, w * patch_size)

    return mask_img
