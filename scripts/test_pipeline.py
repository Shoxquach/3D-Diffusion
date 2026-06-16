"""Quick sanity check for dataset and model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.dataset import ModelNetPointCloudDataset
from models.denoiser import PointCloudDenoiser
from models.diffusion import GaussianDiffusion


def main() -> None:
    dataset = ModelNetPointCloudDataset(
        root="data/ModelNet40",
        category="airplane",
        split="train",
        num_points=2048,
        augment=True,
    )
    print(f"Dataset size: {len(dataset)}")

    x0 = dataset[0]
    print(f"Single sample shape: {tuple(x0.shape)}")
    print(f"Value range: [{x0.min().item():.3f}, {x0.max().item():.3f}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = PointCloudDenoiser(num_points=2048, hidden_dim=128)
    diffusion = GaussianDiffusion(denoiser, timesteps=100).to(device)

    batch = torch.stack([dataset[i] for i in range(4)]).to(device)
    loss = diffusion.training_loss(batch)
    print(f"Training loss (untrained model): {loss.item():.4f}")

    with torch.no_grad():
        sample = diffusion.ddim_sample((1, 2048, 3), device=device, steps=10)
    print(f"Sample shape: {tuple(sample.shape)}")


if __name__ == "__main__":
    main()
