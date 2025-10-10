from __future__ import annotations
import torch
import torch.nn as nn

class ConcatMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([f_left, f_right], dim=-1))

def fuse_mean(f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
    return 0.5 * (f_left + f_right)

def fuse_max(f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
    return torch.maximum(f_left, f_right)
