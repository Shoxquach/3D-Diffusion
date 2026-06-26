"""
Point-E transformer backbone (adapted from openai/point-e/point_e/models/transformer.py).

Unconditional PointDiffusionTransformer only; CLIP conditioning is omitted.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import init_linear, timestep_embedding


class QKVMultiheadAttention(nn.Module):
    def __init__(self, heads: int) -> None:
        super().__init__()
        self.heads = heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(batch, n_ctx, 3, self.heads, attn_ch)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Flash / memory-efficient SDPA; avoids materializing (B, H, T, T) weights.
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(batch, n_ctx, -1)


class MultiheadAttention(nn.Module):
    def __init__(self, n_ctx: int, width: int, heads: int, init_scale: float) -> None:
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_qkv(x)
        x = self.attention(x)
        return self.c_proj(x)


class MLP(nn.Module):
    def __init__(self, width: int, init_scale: float) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_ctx: int, width: int, heads: int, init_scale: float) -> None:
        super().__init__()
        self.attn = MultiheadAttention(n_ctx, width, heads, init_scale)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width, init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        block_init = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(n_ctx, width, heads, block_init)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x


class PointDiffusionTransformer(nn.Module):
    """
    Unconditional point cloud diffusion transformer (Point-E base40M-uncond, simplified).

    Official Point-E uses 6 input / 12 output channels (xyz+rgb, epsilon + variance).
    This project diffuses xyz only (3 / 3 channels) for ModelNet40 shapes without color.

    Tensor layout follows Point-E: input/output are (B, C, N) where N is n_ctx points.
    """

    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 384,
        layers: int = 8,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = True,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width ({width}) must be divisible by heads ({heads})")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond

        time_init = init_scale * math.sqrt(1.0 / width)
        self.time_embed = MLP(width, time_init)
        self.ln_pre = nn.LayerNorm(width)
        self.backbone = Transformer(
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width)
        self.input_proj = nn.Linear(input_channels, width)
        self.output_proj = nn.Linear(width, output_channels)

        init_linear(self.input_proj, init_scale)
        init_linear(self.output_proj, init_scale)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, C, N) noised point cloud.
        :param t: (B,) diffusion timesteps.
        :return: (B, C_out, N) predicted noise (or mean+var if output_channels > C).
        """
        if x.shape[-1] != self.n_ctx:
            raise ValueError(f"Expected n_ctx={self.n_ctx}, got {x.shape[-1]} points")

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self,
        x: torch.Tensor,
        cond_as_token: Sequence[tuple[torch.Tensor, bool]],
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # (B, C, N) -> (B, N, width)

        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]

        extra_tokens = [
            (emb[:, None] if emb.ndim == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        if extra_tokens:
            h = torch.cat(extra_tokens + [h], dim=1)

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)

        if extra_tokens:
            n_cond = sum(token.shape[1] for token in extra_tokens)
            h = h[:, n_cond:]

        h = self.output_proj(h)
        return h.permute(0, 2, 1)  # (B, C_out, N)
