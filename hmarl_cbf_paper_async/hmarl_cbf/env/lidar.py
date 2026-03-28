from __future__ import annotations

from typing import Dict, List

import numpy as np

from hmarl_cbf.types import LidarScan


class LidarModel:
    def __init__(self, n_beam: int = 32, max_range: float = 6.0, noise_std: float = 0.0) -> None:
        if n_beam <= 0:
            raise ValueError("n_beam must be positive")
        if max_range <= 0:
            raise ValueError("max_range must be positive")
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        self.n_beam = n_beam
        self.max_range = max_range
        self.noise_std = noise_std
        self._angles = np.linspace(0.0, 2.0 * np.pi, n_beam, endpoint=False, dtype=np.float32)

    @staticmethod
    def _ray_circle_distance(
        origin: np.ndarray,
        direction: np.ndarray,
        center: np.ndarray,
        radius: float,
        max_range: float,
    ) -> float:
        rel = center - origin
        proj = float(np.dot(rel, direction))
        if proj < 0.0:
            return max_range
        closest_sq = float(np.dot(rel, rel) - proj * proj)
        r_sq = float(radius * radius)
        if closest_sq > r_sq:
            return max_range
        thc = float(np.sqrt(max(r_sq - closest_sq, 0.0)))
        t_hit = proj - thc
        if t_hit < 0.0:
            t_hit = proj + thc
        if t_hit < 0.0 or t_hit > max_range:
            return max_range
        return t_hit

    def scan(
        self,
        origin: np.ndarray,
        obstacles: List[Dict[str, np.ndarray | float]],
        neighbors: List[Dict[str, np.ndarray | float]] | None = None,
    ) -> LidarScan:
        origin = np.asarray(origin, dtype=np.float32).reshape(2)
        ranges = np.full(self.n_beam, self.max_range, dtype=np.float32)

        combined = list(obstacles)
        if neighbors is not None:
            combined.extend(neighbors)

        for i, angle in enumerate(self._angles):
            direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
            best = self.max_range
            for item in combined:
                center = np.asarray(item["center"], dtype=np.float32).reshape(2)
                radius = float(item["radius"])
                dist = self._ray_circle_distance(origin, direction, center, radius, self.max_range)
                if dist < best:
                    best = dist
            ranges[i] = best

        if self.noise_std > 0:
            ranges = ranges + np.random.normal(0.0, self.noise_std, size=ranges.shape).astype(np.float32)
            ranges = np.clip(ranges, 0.0, self.max_range)

        return LidarScan(ranges=ranges, max_range=self.max_range, noise_std=self.noise_std)
