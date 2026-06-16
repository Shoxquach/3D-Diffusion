"""DDPM diffusion process for point clouds."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-8, 0.999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
    ) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def _extract(self, values: torch.Tensor, t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        batch = t.shape[0]
        out = values.gather(-1, t)
        return out.reshape(batch, *((1,) * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = self.model(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        batch_size = x_t.shape[0]
        t_batch = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        pred_noise = self.model(x_t, t_batch)

        beta_t = self.betas[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        model_mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * pred_noise)

        if t == 0:
            return model_mean

        noise = torch.randn_like(x_t)
        posterior_variance_t = self.posterior_variance[t]
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t)
        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()

        for i, t in enumerate(times):
            t_batch = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)
            pred_noise = self.model(x, t_batch)

            alpha_bar_t = self.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

            if i == len(times) - 1:
                x = x0_pred
                break

            t_prev = times[i + 1]
            alpha_bar_prev = self.alphas_cumprod[t_prev]

            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
                * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_noise
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

        return x
