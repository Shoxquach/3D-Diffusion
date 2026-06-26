"""Training script for point cloud diffusion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ModelNetPointCloudDataset
from models.denoiser import build_denoiser
from models.diffusion import GaussianDiffusion

_RESUME_KEYS = ("denoiser_type", "num_points", "timesteps", "beta_schedule", "hidden_dim")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train point cloud diffusion model")
    parser.add_argument("--config", type=str, default="configs/airplane.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from",
    )
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    cfg: dict,
    diffusion: GaussianDiffusion,
    optimizer: torch.optim.Optimizer,
    loss: float,
    global_step: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "config": cfg,
            "model_state_dict": diffusion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_resume_checkpoint(
    path: Path,
    cfg: dict,
    diffusion: GaussianDiffusion,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    device: torch.device,
) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    for key in _RESUME_KEYS:
        if ckpt_cfg.get(key) != cfg.get(key):
            raise ValueError(
                f"Config mismatch on '{key}': checkpoint={ckpt_cfg.get(key)!r}, "
                f"config={cfg.get(key)!r}"
            )

    diffusion.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    last_epoch = int(ckpt["epoch"])
    global_step = int(ckpt.get("global_step", last_epoch * steps_per_epoch))
    return last_epoch + 1, global_step


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

    if device.type == "cuda":
        if cfg.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
        if cfg.get("tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print(f"Loading dataset: {cfg['category']}/{cfg['split']}")
    dataset = ModelNetPointCloudDataset(
        root=cfg["data_root"],
        category=cfg["category"],
        split=cfg["split"],
        num_points=cfg["num_points"],
        augment=True,
        preload=cfg.get("preload_data", False),
    )
    num_workers = cfg.get("num_workers")
    if num_workers is None:
        # Windows spawn duplicates the full Python/PyTorch stack per worker (multi-GB RAM).
        num_workers = 0 if sys.platform == "win32" else 2
    if num_workers > 0 and cfg.get("preload_data", False):
        print("Note: preload_data disabled when num_workers > 0 (avoids copying dataset per worker).")
        dataset._preloaded = None

    pin_memory = cfg.get("pin_memory", True) and device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    if num_workers == 0:
        print(f"DataLoader: num_workers=0 (in-memory preload={'on' if dataset._preloaded else 'off'})")
    else:
        print(f"DataLoader: num_workers={num_workers}")

    denoiser = build_denoiser(cfg)
    param_count = sum(p.numel() for p in denoiser.parameters())
    print(f"Denoiser: transformer ({param_count:,} parameters)")
    diffusion = GaussianDiffusion(
        model=denoiser,
        timesteps=cfg["timesteps"],
        beta_schedule=cfg["beta_schedule"],
    ).to(device)

    use_amp = cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("Mixed precision (AMP): enabled")

    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_resume_checkpoint(
            Path(args.resume),
            cfg,
            diffusion,
            optimizer,
            steps_per_epoch=len(loader),
            device=device,
        )
        if start_epoch > cfg["epochs"]:
            print(
                f"Checkpoint already at epoch {start_epoch - 1}, "
                f"which meets or exceeds target epochs ({cfg['epochs']}). Nothing to do."
            )
            return
        print(
            f"Resumed from {args.resume}: starting at epoch {start_epoch}/{cfg['epochs']} "
            f"(global_step={global_step})"
        )

    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    if grad_accum_steps > 1:
        print(f"Gradient accumulation: {grad_accum_steps} steps (effective batch={cfg['batch_size'] * grad_accum_steps})")

    avg_loss = 0.0
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        diffusion.train()
        epoch_loss = 0.0
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{cfg['epochs']}",
            dynamic_ncols=True,
            leave=False,
        )

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            x0 = batch.to(device, non_blocking=pin_memory)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion.training_loss(x0) / grad_accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss += loss.item() * grad_accum_steps
            pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}", refresh=False)

        pbar.close()
        avg_loss = epoch_loss / len(loader)
        tqdm.write(f"Epoch {epoch} average loss: {avg_loss:.6f}")

        if epoch % cfg.get("save_every", 50) == 0 or epoch == cfg["epochs"]:
            ckpt_path = save_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                ckpt_path,
                epoch=epoch,
                cfg=cfg,
                diffusion=diffusion,
                optimizer=optimizer,
                loss=avg_loss,
                global_step=global_step,
            )
            tqdm.write(f"Saved checkpoint: {ckpt_path}")

            latest_path = save_dir / "latest.pt"
            save_checkpoint(
                latest_path,
                epoch=epoch,
                cfg=cfg,
                diffusion=diffusion,
                optimizer=optimizer,
                loss=avg_loss,
                global_step=global_step,
            )

    tqdm.write(f"Training finished. Final checkpoint: {save_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
