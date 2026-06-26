"""ModelNet40 point cloud dataset for diffusion training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
        preload: bool = False,
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

        self._preloaded: list[np.ndarray] | None = None
        self._ensure_cache()

        if preload:
            total = len(self.mesh_paths)
            print(f"Preloading {total} point clouds ({self.num_points} pts) into memory...")
            self._preloaded = [
                np.load(self._cache_path(mesh_path)).astype(np.float32)
                for mesh_path in tqdm(self.mesh_paths, desc="Preload", unit="mesh")
            ]
            print("Preload complete.")

    def _cache_path(self, mesh_path: Path) -> Path:
        name = mesh_path.stem + f"_{self.num_points}.npy"
        return self.cache_dir / name

    def _cache_status(self) -> tuple[int, int]:
        cached = sum(1 for mesh_path in self.mesh_paths if self._cache_path(mesh_path).exists())
        return cached, len(self.mesh_paths)

    def _ensure_cache(self) -> None:
        """Build missing .npy caches by sampling OFF meshes, with progress output."""
        cached, total = self._cache_status()
        cache_glob = f"*_{self.num_points}.npy"

        if cached == total:
            print(
                f"Point cloud cache: {cached}/{total} ready "
                f"(num_points={self.num_points}, dir={self.cache_dir})"
            )
            return

        missing = [mesh_path for mesh_path in self.mesh_paths if not self._cache_path(mesh_path).exists()]
        print(
            f"Point cloud cache: {cached}/{total} ready for num_points={self.num_points}\n"
            f"  cache dir: {self.cache_dir}\n"
            f"  need to sample {len(missing)} mesh(es) from OFF (pattern: {cache_glob})"
        )
        for mesh_path in tqdm(missing, desc="Sample/cache", unit="mesh"):
            self._sample_and_save(mesh_path)

        print(f"Cache build complete: {total} files at {self.num_points} points -> {self.cache_dir}")

    def _sample_and_save(self, mesh_path: Path) -> np.ndarray:
        cache_path = self._cache_path(mesh_path)
        mesh = load_mesh(str(mesh_path))
        points = sample_point_cloud(mesh, self.num_points)
        np.save(cache_path, points)
        return points

    def _load_or_create_points(self, mesh_path: Path) -> np.ndarray:
        cache_path = self._cache_path(mesh_path)
        if cache_path.exists():
            return np.load(cache_path)

        print(
            f"Cache miss, sampling on demand: {mesh_path.name} -> {cache_path.name} "
            f"({self.num_points} points)"
        )
        return self._sample_and_save(mesh_path)

    def __len__(self) -> int:
        return len(self.mesh_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._preloaded is not None:
            points = self._preloaded[index].copy()
        else:
            mesh_path = self.mesh_paths[index]
            points = self._load_or_create_points(mesh_path).astype(np.float32)

        if self.augment:
            points = random_rotate_y(points)
            points = random_scale(points)

        return torch.from_numpy(points)
