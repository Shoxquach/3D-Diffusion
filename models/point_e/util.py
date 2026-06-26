"""Utilities ported from openai/point-e (point_e/models/util.py)."""

from __future__ import annotations

import math

import torch


def init_linear(layer: torch.nn.Linear, stddev: float) -> None:
    torch.nn.init.normal_(layer.weight, std=stddev)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, 0.0)


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings (cos then sin), matching Point-E / OpenAI style.

    :param timesteps: (N,) indices, one per batch element.
    :return: (N, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
