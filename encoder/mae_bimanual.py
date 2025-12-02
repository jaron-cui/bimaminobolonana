"""
Bimanual Cross-Attention MAE with CLIP backbone.

Main model for pretraining visual encoder on bimanual manipulation data.
Inspired by "Touch in the Wild" cross-modal attention approach.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import MultiViewEncoder
from .mae_clip import (
    CLIPPatchExtractor,
    BimanualCrossAttention,
    MAEDecoder,
    patchify,
    compute_mae_loss,
)


class BimanualCrossMAE(MultiViewEncoder):
    """
    MAE with CLIP backbone and cross-attention between bimanual views.

    Architecture:
    1. CLIP ViT encoder extracts patch tokens from both views
    2. Random masking (75%) applied independently to each view
    3. Cross-attention fuses information between visible tokens
    4. Separate decoders reconstruct each view

    For downstream policy:
    - Use forward_single() or forward_pair() without masking
    - Returns 512-dim features suitable for ACT policy
    """

    def __init__(
        self,
        clip_model: str = "ViT-B-32",
        pretrained: str = "openai",
        out_dim: int = 512,
        freeze_clip: bool = False,
        # Cross attention
        cross_attn_layers: int = 2,
        cross_attn_heads: int = 12,
        # Decoder
        decoder_dim: int = 256,
        decoder_depth: int = 4,
        decoder_heads: int = 4,
        # MAE
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        cross_view_mode: bool = False,  # Cross-view completion mode
        # Fusion for policy
        fuse: str = "mean",
    ):
        """
        Args:
            clip_model: CLIP model variant (e.g., "ViT-B-32", "ViT-B-16")
            pretrained: Pretrained weights source (e.g., "openai")
            out_dim: Output feature dimension
            freeze_clip: Whether to freeze CLIP backbone
            cross_attn_layers: Number of cross-attention layers
            cross_attn_heads: Number of attention heads
            decoder_dim: Decoder hidden dimension
            decoder_depth: Number of decoder layers
            decoder_heads: Number of decoder attention heads
            mask_ratio: Fraction of patches to mask (0-1)
            norm_pix_loss: Normalize target pixels by variance
            cross_view_mode: If True, use cross-view completion:
                            - Randomly mask ONE view (left or right)
                            - Keep the other view fully visible
                            - Reconstruct only the masked view using cross-attention
                            This forces learning of cross-view correspondence.
            fuse: Fusion method for left/right features
        """
        super().__init__(out_dim=out_dim, fuse=fuse)

        # CLIP encoder (shared for both views)
        self.clip_extractor = CLIPPatchExtractor(
            model_name=clip_model,
            pretrained=pretrained,
            freeze=freeze_clip,
        )

        self.hidden_dim = self.clip_extractor.hidden_dim  # 768 for ViT-B
        self.patch_size = self.clip_extractor.patch_size  # 32 for ViT-B-32

        # Calculate number of patches for 224x224 images
        self.img_size = 224
        self.num_patches = (self.img_size // self.patch_size) ** 2  # 49 for 7x7

        # Cross attention between left and right tokens
        self.cross_attention = BimanualCrossAttention(
            dim=self.hidden_dim,
            num_heads=cross_attn_heads,
            num_layers=cross_attn_layers,
        )

        # Separate decoders for left and right reconstruction
        self.decoder_left = MAEDecoder(
            encoder_dim=self.hidden_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            num_patches=self.num_patches,
            patch_size=self.patch_size,
        )
        self.decoder_right = MAEDecoder(
            encoder_dim=self.hidden_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            num_patches=self.num_patches,
            patch_size=self.patch_size,
        )

        # Output projection for policy (hidden_dim -> out_dim)
        self.proj = nn.Linear(self.hidden_dim, out_dim)

        # MAE parameters
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.cross_view_mode = cross_view_mode

    def forward_mae(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        MAE forward pass with cross-attention for training.

        Args:
            left_img: (B, 3, 224, 224) left camera image
            right_img: (B, 3, 224, 224) right camera image

        Returns:
            dict with:
                - loss: reconstruction loss
                - pred_left, pred_right: (B, N, patch_size^2 * 3) predictions
                - mask_left, mask_right: masks (1=masked, 0=visible)
                - masked_view: 'left', 'right', or 'both' (for cross_view_mode)
        """
        if self.cross_view_mode:
            return self._forward_cross_view(left_img, right_img)
        else:
            return self._forward_both_views(left_img, right_img)

    def _forward_cross_view(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Cross-View Completion: mask ONE view, use the other to reconstruct.
        
        Randomly choose to mask left or right view each forward pass.
        The visible view provides full context through cross-attention.
        """
        B = left_img.shape[0]
        device = left_img.device
        
        # Randomly choose which view to mask (per batch, same for all samples)
        mask_left_view = torch.rand(1).item() < 0.5
        
        if mask_left_view:
            # LEFT is masked, RIGHT is fully visible
            # Get masked tokens for left
            left_vis, left_restore, left_mask = self.clip_extractor.get_patch_tokens_with_mask(
                left_img, self.mask_ratio
            )
            # Get ALL tokens for right (no masking)
            right_tokens, _ = self.clip_extractor.get_patch_tokens(right_img)
            right_mask = torch.zeros(B, self.num_patches, device=device)  # No masking
            
            # Cross attention: masked left attends to full right
            left_fused, right_fused = self.cross_attention(left_vis, right_tokens)
            
            # Only decode left (the masked view)
            pred_left = self.decoder_left(left_fused, left_restore)
            
            # For consistency, also decode right (but no loss)
            # We need restore indices for right - since no masking, just use identity
            right_restore = torch.arange(self.num_patches, device=device).unsqueeze(0).expand(B, -1)
            pred_right = self.decoder_right(right_fused, right_restore)
            
            # Compute loss ONLY on masked view
            target_left = patchify(left_img, self.patch_size)
            loss = compute_mae_loss(pred_left, target_left, left_mask, self.norm_pix_loss)
            
            masked_view = "left"
            
        else:
            # RIGHT is masked, LEFT is fully visible
            # Get ALL tokens for left (no masking)
            left_tokens, _ = self.clip_extractor.get_patch_tokens(left_img)
            left_mask = torch.zeros(B, self.num_patches, device=device)  # No masking
            # Get masked tokens for right
            right_vis, right_restore, right_mask = self.clip_extractor.get_patch_tokens_with_mask(
                right_img, self.mask_ratio
            )
            
            # Cross attention: full left, masked right
            left_fused, right_fused = self.cross_attention(left_tokens, right_vis)
            
            # Only decode right (the masked view)
            pred_right = self.decoder_right(right_fused, right_restore)
            
            # For consistency, also decode left
            left_restore = torch.arange(self.num_patches, device=device).unsqueeze(0).expand(B, -1)
            pred_left = self.decoder_left(left_fused, left_restore)
            
            # Compute loss ONLY on masked view
            target_right = patchify(right_img, self.patch_size)
            loss = compute_mae_loss(pred_right, target_right, right_mask, self.norm_pix_loss)
            
            masked_view = "right"
        
        return {
            "loss": loss,
            "pred_left": pred_left,
            "pred_right": pred_right,
            "mask_left": left_mask,
            "mask_right": right_mask,
            "masked_view": masked_view,
        }

    def _forward_both_views(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Standard MAE: mask BOTH views independently, reconstruct both.
        """
        # Mask both views
        left_vis, left_restore, left_mask = self.clip_extractor.get_patch_tokens_with_mask(
            left_img, self.mask_ratio
        )
        right_vis, right_restore, right_mask = self.clip_extractor.get_patch_tokens_with_mask(
            right_img, self.mask_ratio
        )

        # Cross attention between visible tokens of both views
        left_fused, right_fused = self.cross_attention(left_vis, right_vis)

        # Decode both views
        pred_left = self.decoder_left(left_fused, left_restore)
        pred_right = self.decoder_right(right_fused, right_restore)

        # Compute targets (patchified images)
        target_left = patchify(left_img, self.patch_size)
        target_right = patchify(right_img, self.patch_size)

        # Compute losses on both views
        loss_left = compute_mae_loss(pred_left, target_left, left_mask, self.norm_pix_loss)
        loss_right = compute_mae_loss(pred_right, target_right, right_mask, self.norm_pix_loss)

        loss = (loss_left + loss_right) / 2

        return {
            "loss": loss,
            "loss_left": loss_left,
            "loss_right": loss_right,
            "pred_left": pred_left,
            "pred_right": pred_right,
            "mask_left": left_mask,
            "mask_right": right_mask,
            "masked_view": "both",
        }

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode single image for inference (no masking).

        Args:
            x: (B, 3, 224, 224) input image

        Returns:
            features: (B, out_dim) image features
        """
        patch_tokens, cls_token = self.clip_extractor.get_patch_tokens(x)

        # Use mean pooling over patch tokens
        features = patch_tokens.mean(dim=1)  # (B, hidden_dim)

        # Project to output dimension
        features = self.proj(features)  # (B, out_dim)

        return features

    def forward_pair(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode image pair with cross-attention for inference.

        Args:
            left_img: (B, 3, 224, 224) left camera image
            right_img: (B, 3, 224, 224) right camera image

        Returns:
            dict with 'left', 'right', 'fused' features
        """
        # Get patch tokens without masking
        left_tokens, _ = self.clip_extractor.get_patch_tokens(left_img)
        right_tokens, _ = self.clip_extractor.get_patch_tokens(right_img)

        # Cross attention
        left_fused, right_fused = self.cross_attention(left_tokens, right_tokens)

        # Pool to get features
        left_feat = self.proj(left_fused.mean(dim=1))  # (B, out_dim)
        right_feat = self.proj(right_fused.mean(dim=1))  # (B, out_dim)

        # Fuse
        fused = self._fuse(left_feat, right_feat)

        return {
            "left": left_feat,
            "right": right_feat,
            "fused": fused,
        }

    def get_reconstruction(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get reconstructed images for visualization.

        Returns original images, masked images, and reconstructions.
        If pixel_mask_ratio is set, shows sparse pixel-level masking pattern.
        """
        with torch.no_grad():
            result = self.forward_mae(left_img, right_img)

            h = w = self.img_size // self.patch_size
            B = left_img.shape[0]

            # Unpatchify predictions
            recon_left = self.decoder_left.unpatchify(result["pred_left"], h, w)
            recon_right = self.decoder_right.unpatchify(result["pred_right"], h, w)

            if self.pixel_mask_ratio is not None and "pixel_mask_left" in result:
                # Use sparse pixel-level mask for visualization
                # Convert from (B, N, D) to (B, 3, H, W)
                mask_left = pixel_mask_to_image(
                    result["pixel_mask_left"], self.patch_size, self.img_size
                )
                mask_right = pixel_mask_to_image(
                    result["pixel_mask_right"], self.patch_size, self.img_size
                )
            else:
                # Original patch-level mask visualization
                patch_mask_left = result["mask_left"]
                patch_mask_right = result["mask_right"]

                mask_left = patch_mask_left.reshape(-1, h, w)
                mask_right = patch_mask_right.reshape(-1, h, w)

                # Expand masks to image size
                mask_left = mask_left.unsqueeze(1).repeat(1, 3, 1, 1)
                mask_left = mask_left.repeat_interleave(self.patch_size, dim=2)
                mask_left = mask_left.repeat_interleave(self.patch_size, dim=3)

                mask_right = mask_right.unsqueeze(1).repeat(1, 3, 1, 1)
                mask_right = mask_right.repeat_interleave(self.patch_size, dim=2)
                mask_right = mask_right.repeat_interleave(self.patch_size, dim=3)

            return {
                "original_left": left_img,
                "original_right": right_img,
                "masked_left": left_img * (1 - mask_left),
                "masked_right": right_img * (1 - mask_right),
                "recon_left": recon_left,
                "recon_right": recon_right,
                "mask_left": mask_left,
                "mask_right": mask_right,
            }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "BimanualCrossMAE":
        """
        Load pretrained model from checkpoint.

        Args:
            checkpoint_path: path to .pt checkpoint file
            **kwargs: override config parameters

        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Get config from checkpoint
        config = checkpoint.get("config", {})
        config.update(kwargs)

        # Create model
        model = cls(**config)

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    def save_pretrained(self, checkpoint_path: str, config: dict = None):
        """
        Save model checkpoint.

        Args:
            checkpoint_path: path to save .pt file
            config: optional config dict to save
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": config or {},
        }, checkpoint_path)

    def get_encoder_for_policy(self) -> nn.Module:
        """
        Get the encoder module for downstream policy training.

        Returns a module that takes (left_img, right_img) and returns fused features.
        """
        return _BimanualEncoderForPolicy(self)


class _BimanualEncoderForPolicy(nn.Module):
    """
    Wrapper for using pretrained BimanualCrossMAE in policy.
    """

    def __init__(self, mae_model: BimanualCrossMAE):
        super().__init__()
        self.mae = mae_model

    def forward(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns fused features from both views.
        """
        result = self.mae.forward_pair(left_img, right_img)
        return result["fused"]

    @property
    def out_dim(self) -> int:
        return self.mae.out_dim
