"""Training script for point cloud diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ModelNetPointCloudDataset
from models.denoiser import PointCloudDenoiser
from models.diffusion import GaussianDiffusion


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train point cloud diffusion model")
    parser.add_argument("--config", type=str, default="configs/airplane.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ModelNetPointCloudDataset(
        root=cfg["data_root"],
        category=cfg["category"],
        split=cfg["split"],
        num_points=cfg["num_points"],
        augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        drop_last=True,
    )

    denoiser = PointCloudDenoiser(
        num_points=cfg["num_points"],
        hidden_dim=cfg["hidden_dim"],
    )
    diffusion = GaussianDiffusion(
        model=denoiser,
        timesteps=cfg["timesteps"],
        beta_schedule=cfg["beta_schedule"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        diffusion.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}")

        for batch in pbar:
            x0 = batch.to(device)
            optimizer.zero_grad()
            loss = diffusion.training_loss(x0)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % cfg.get("log_every", 20) == 0:
                pbar.set_description(f"Epoch {epoch} | loss {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} average loss: {avg_loss:.6f}")

        if epoch % cfg.get("save_every", 50) == 0 or epoch == cfg["epochs"]:
            ckpt_path = save_dir / f"epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "config": cfg,
                    "model_state_dict": diffusion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = save_dir / "latest.pt"
    torch.save(
        {
            "epoch": cfg["epochs"],
            "config": cfg,
            "model_state_dict": diffusion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        final_path,
    )
    print(f"Training finished. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
