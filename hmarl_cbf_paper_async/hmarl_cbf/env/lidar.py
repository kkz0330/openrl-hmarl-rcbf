from __future__ import annotations

from typing import Dict, List

import numpy as np

from hmarl_cbf.env.obstacles import ray_obstacle_distance
from hmarl_cbf.types import (
    LIDAR_HIT_NEIGHBOR,
    LIDAR_HIT_NONE,
    LIDAR_HIT_OBSTACLE,
    LidarScan,
)


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

    def scan(
        self,
        origin: np.ndarray,
        obstacles: List[Dict[str, np.ndarray | float]],
        neighbors: List[Dict[str, np.ndarray | float]] | None = None,
    ) -> LidarScan:
        origin = np.asarray(origin, dtype=np.float32).reshape(2)
        ranges = np.full(self.n_beam, self.max_range, dtype=np.float32)
        hit_points = np.zeros((self.n_beam, 2), dtype=np.float32)
        hit_valid = np.zeros((self.n_beam,), dtype=np.bool_)
        hit_kinds = np.full((self.n_beam,), LIDAR_HIT_NONE, dtype=np.int32)

        typed_items: List[tuple[int, Dict[str, np.ndarray | float]]] = []
        typed_items.extend((LIDAR_HIT_OBSTACLE, obs) for obs in list(obstacles))
        if neighbors is not None:
            typed_items.extend((LIDAR_HIT_NEIGHBOR, nbr) for nbr in list(neighbors))

        for i, angle in enumerate(self._angles):
            direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
            best = self.max_range
            best_kind = LIDAR_HIT_NONE
            for item_kind, item in typed_items:
                dist = ray_obstacle_distance(origin=origin, direction=direction, item=item, max_range=self.max_range)
                if dist < best:
                    best = dist
                    best_kind = int(item_kind)
            ranges[i] = best
            hit_points[i] = (origin + best * direction).astype(np.float32)
            is_valid = bool(best < self.max_range - 1e-6)
            hit_valid[i] = is_valid
            hit_kinds[i] = int(best_kind if is_valid else LIDAR_HIT_NONE)

        if self.noise_std > 0:
            ranges = ranges + np.random.normal(0.0, self.noise_std, size=ranges.shape).astype(np.float32)
            ranges = np.clip(ranges, 0.0, self.max_range)
            directions = np.stack(
                [np.cos(self._angles).astype(np.float32), np.sin(self._angles).astype(np.float32)],
                axis=1,
            )
            hit_points = origin.reshape(1, 2) + ranges.reshape(-1, 1) * directions
            hit_valid = ranges < (self.max_range - 1e-6)
            hit_kinds = np.where(hit_valid, hit_kinds, LIDAR_HIT_NONE).astype(np.int32)

        return LidarScan(
            ranges=ranges,
            max_range=self.max_range,
            angles=self._angles.copy(),
            origin=origin.copy(),
            hit_points=hit_points.astype(np.float32),
            hit_valid=hit_valid.astype(np.bool_),
            hit_kinds=hit_kinds.astype(np.int32),
            noise_std=self.noise_std,
        )
