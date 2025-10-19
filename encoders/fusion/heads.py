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

class GatedFusion(nn.Module):
    """Learn a gate g in (0,1): fused = g * f_left + (1 - g) * f_right"""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([f_left, f_right], dim=-1))  # [B,1]
        return g * f_left + (1.0 - g) * f_right

class BilinearFusion(nn.Module):
    """Hadamard + MLP: fused = MLP([f_left ⊙ f_right; f_left; f_right])"""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
        had = f_left * f_right
        x = torch.cat([had, f_left, f_right], dim=-1)
        return self.net(x)

def fuse_mean(f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
    return 0.5 * (f_left + f_right)

def fuse_max(f_left: torch.Tensor, f_right: torch.Tensor) -> torch.Tensor:
    return torch.maximum(f_left, f_right)
