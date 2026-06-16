"""Mesh loading and point cloud sampling utilities."""

from __future__ import annotations

import numpy as np
import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    center = points.mean(axis=0)
    points = points - center
    scale = np.linalg.norm(points, axis=1).max()
    if scale < 1e-8:
        return points
    return points / scale


def farthest_point_sample(points: np.ndarray, num_points: int) -> np.ndarray:
    """Iterative farthest point sampling."""
    n = points.shape[0]
    if n <= num_points:
        if n == num_points:
            return points
        idx = np.random.choice(n, num_points, replace=True)
        return points[idx]

    selected = np.zeros(num_points, dtype=np.int64)
    distances = np.full(n, np.inf, dtype=np.float32)
    farthest = np.random.randint(0, n)

    for i in range(num_points):
        selected[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))

    return points[selected]


def sample_point_cloud(
    mesh: trimesh.Trimesh,
    num_points: int,
    oversample: int = 8192,
) -> np.ndarray:
    """Sample uniform surface points, then FPS to fixed count."""
    count = max(oversample, num_points * 2)
    points, _ = trimesh.sample.sample_surface(mesh, count=count)
    points = farthest_point_sample(points, num_points)
    return normalize_point_cloud(points)


def random_rotate_y(points: np.ndarray) -> np.ndarray:
    angle = np.random.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    return points @ rot.T


def random_scale(points: np.ndarray, low: float = 0.9, high: float = 1.1) -> np.ndarray:
    scale = np.random.uniform(low, high)
    return points * scale
