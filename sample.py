"""Sample point clouds from a trained diffusion model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models.denoiser import PointCloudDenoiser
from models.diffusion import GaussianDiffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample point clouds from diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--output_dir", type=str, default="outputs/samples")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="ddim", choices=["ddim", "ddpm"])
    return parser.parse_args()


def save_ply(path: Path, points: np.ndarray) -> None:
    """Save point cloud as ASCII PLY without external dependencies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    denoiser = PointCloudDenoiser(
        num_points=cfg["num_points"],
        hidden_dim=cfg["hidden_dim"],
    )
    diffusion = GaussianDiffusion(
        model=denoiser,
        timesteps=cfg["timesteps"],
        beta_schedule=cfg["beta_schedule"],
    ).to(device)
    diffusion.load_state_dict(ckpt["model_state_dict"])
    diffusion.eval()

    shape = (args.num_samples, cfg["num_points"], 3)
    print(f"Sampling {args.num_samples} point clouds with {args.method} ({args.steps} steps)...")

    if args.method == "ddim":
        samples = diffusion.ddim_sample(shape, device=device, steps=args.steps)
    else:
        samples = diffusion.sample(shape, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_np = samples.cpu().numpy()
    for i in range(args.num_samples):
        npy_path = output_dir / f"sample_{i:03d}.npy"
        ply_path = output_dir / f"sample_{i:03d}.ply"
        np.save(npy_path, samples_np[i])
        save_ply(ply_path, samples_np[i])
        print(f"Saved {npy_path} and {ply_path}")

    print("Done.")


if __name__ == "__main__":
    main()
