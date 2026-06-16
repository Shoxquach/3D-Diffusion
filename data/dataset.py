"""ModelNet40 point cloud dataset for diffusion training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.mesh import load_mesh, random_rotate_y, random_scale, sample_point_cloud


class ModelNetPointCloudDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        category: str = "airplane",
        split: str = "train",
        num_points: int = 2048,
        augment: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.num_points = num_points
        self.augment = augment

        mesh_dir = self.root / category / split
        if not mesh_dir.exists():
            raise FileNotFoundError(f"Dataset path not found: {mesh_dir}")

        self.mesh_paths = sorted(mesh_dir.glob("*.off"))
        if not self.mesh_paths:
            raise FileNotFoundError(f"No .off files found in {mesh_dir}")

        self.cache_dir = Path(cache_dir) if cache_dir else self.root / "_cache" / category / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, mesh_path: Path) -> Path:
        name = mesh_path.stem + f"_{self.num_points}.npy"
        return self.cache_dir / name

    def _load_or_create_points(self, mesh_path: Path) -> np.ndarray:
        cache_path = self._cache_path(mesh_path)
        if cache_path.exists():
            return np.load(cache_path)

        mesh = load_mesh(str(mesh_path))
        points = sample_point_cloud(mesh, self.num_points)
        np.save(cache_path, points)
        return points

    def __len__(self) -> int:
        return len(self.mesh_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        mesh_path = self.mesh_paths[index]
        points = self._load_or_create_points(mesh_path)

        if self.augment:
            points = random_rotate_y(points)
            points = random_scale(points)

        return torch.from_numpy(points.astype(np.float32))
