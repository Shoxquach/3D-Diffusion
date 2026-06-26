"""Point cloud denoisers for diffusion models."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.point_e.transformer import PointDiffusionTransformer


class PointCloudTransformerDenoiser(nn.Module):
    """
    Adapter around Point-E PointDiffusionTransformer.

    Exposes the project convention (B, N, C) while the core model uses (B, C, N).
    Default width/layers are reduced vs official base40M-uncond (512 width, 12 layers).
    """

    def __init__(
        self,
        num_points: int = 1024,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = True,
        input_channels: int = 3,
        output_channels: int = 3,
        **_unused: object,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.core = PointDiffusionTransformer(
            input_channels=input_channels,
            output_channels=output_channels,
            n_ctx=num_points,
            width=hidden_dim,
            layers=num_layers,
            heads=num_heads,
            init_scale=init_scale,
            time_token_cond=time_token_cond,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) -> core (B, C, N) -> (B, N, C)
        out = self.core(x.permute(0, 2, 1), t)
        return out.permute(0, 2, 1)


def build_denoiser(cfg: dict) -> nn.Module:
    """Construct the Point-E transformer denoiser from training / sampling config."""
    denoiser_type = cfg.get("denoiser_type", "transformer")
    if denoiser_type not in ("transformer", "point_e"):
        raise ValueError(
            f"Unsupported denoiser_type: {denoiser_type!r}. "
            "Only 'transformer' (Point-E PointDiffusionTransformer) is available."
        )

    return PointCloudTransformerDenoiser(
        num_points=cfg["num_points"],
        hidden_dim=cfg.get("hidden_dim", 256),
        num_layers=cfg.get("num_layers", 6),
        num_heads=cfg.get("num_heads", 8),
        init_scale=cfg.get("init_scale", 0.25),
        time_token_cond=cfg.get("time_token_cond", True),
        input_channels=cfg.get("input_channels", 3),
        output_channels=cfg.get("output_channels", 3),
    )
