"""Simple permutation-equivariant point cloud denoiser."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class PointCloudDenoiser(nn.Module):
    """PointNet-style denoiser with global feature fusion."""

    def __init__(
        self,
        num_points: int = 2048,
        hidden_dim: int = 256,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_points = num_points

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.point_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3), t: (B,)
        time_emb = self.time_mlp(t)
        time_emb = time_emb.unsqueeze(1).expand(-1, x.shape[1], -1)

        point_feat = self.point_encoder(x)
        global_feat = point_feat.max(dim=1, keepdim=True).values.expand_as(point_feat)

        fused = self.fusion(torch.cat([point_feat + time_emb, global_feat], dim=-1))
        return self.output_head(fused)
