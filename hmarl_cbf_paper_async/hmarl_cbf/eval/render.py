from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon
except ImportError:  # pragma: no cover - optional backend
    plt = None  # type: ignore[assignment]
    FuncAnimation = None  # type: ignore[assignment]
    PillowWriter = None  # type: ignore[assignment]
    Polygon = None  # type: ignore[assignment]

from hmarl_cbf.env.obstacles import normalize_obstacle, obstacle_corners


@dataclass(slots=True)
class EpisodeTrace:
    positions: np.ndarray  # shape [T, N, 2]
    goals: np.ndarray  # shape [N, 2]
    obstacles: List[Dict[str, np.ndarray | float]]
    unsafe_flags: np.ndarray | None = None  # shape [T, N] (optional)
    frame_labels: List[str] | None = None  # optional text label for each frame


class TrajectoryRenderer:
    def __init__(self, world_size: float, figsize: tuple[float, float] = (6.5, 6.5)) -> None:
        self.world_size = float(world_size)
        self.figsize = figsize

    def _check_backend(self) -> None:
        if plt is None:
            raise RuntimeError("matplotlib is required for rendering")

    def render_static(self, trace: EpisodeTrace, output_path: str | Path) -> str:
        self._check_backend()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        pos = np.asarray(trace.positions, dtype=np.float32)
        goals = np.asarray(trace.goals, dtype=np.float32)
        t, n, _ = pos.shape
        assert goals.shape == (n, 2)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(-self.world_size, self.world_size)
        ax.set_ylim(-self.world_size, self.world_size)
        ax.set_aspect("equal")
        ax.set_title(f"Trajectory (T={t}, N={n})")

        for obs in trace.obstacles:
            obs_norm = normalize_obstacle(obs)
            center = np.asarray(obs_norm["center"], dtype=np.float32).reshape(2)
            if obs_norm["type"] in ("circle", "point"):
                radius = float(obs_norm["radius"])
                patch = plt.Circle((float(center[0]), float(center[1])), radius, color="gray", alpha=0.35)
            else:
                patch = Polygon(obstacle_corners(obs_norm), closed=True, color="gray", alpha=0.35)
            ax.add_patch(patch)

        cmap = plt.get_cmap("tab10")
        for i in range(n):
            color = cmap(i % 10)
            ax.plot(pos[:, i, 0], pos[:, i, 1], color=color, linewidth=1.8)
            ax.scatter(pos[0, i, 0], pos[0, i, 1], color=color, marker="o", s=20)
            ax.scatter(pos[-1, i, 0], pos[-1, i, 1], color=color, marker="x", s=40)
            ax.scatter(goals[i, 0], goals[i, 1], color=color, marker="*", s=65)
        if trace.frame_labels:
            ax.text(
                0.02,
                0.98,
                str(trace.frame_labels[-1]),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 2.0},
            )

        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        return str(output)

    def render_gif(self, trace: EpisodeTrace, output_path: str | Path, fps: int = 8) -> str:
        self._check_backend()
        if FuncAnimation is None or PillowWriter is None:
            raise RuntimeError("matplotlib animation backend unavailable")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        pos = np.asarray(trace.positions, dtype=np.float32)
        goals = np.asarray(trace.goals, dtype=np.float32)
        unsafe = None if trace.unsafe_flags is None else np.asarray(trace.unsafe_flags, dtype=bool)
        frame_labels = list(trace.frame_labels) if trace.frame_labels is not None else None
        t, n, _ = pos.shape
        assert goals.shape == (n, 2)
        if frame_labels is not None and len(frame_labels) < t:
            frame_labels = frame_labels + [""] * (t - len(frame_labels))

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(-self.world_size, self.world_size)
        ax.set_ylim(-self.world_size, self.world_size)
        ax.set_aspect("equal")
        ax.set_title("Episode Animation")

        for obs in trace.obstacles:
            obs_norm = normalize_obstacle(obs)
            center = np.asarray(obs_norm["center"], dtype=np.float32).reshape(2)
            if obs_norm["type"] in ("circle", "point"):
                radius = float(obs_norm["radius"])
                patch = plt.Circle((float(center[0]), float(center[1])), radius, color="gray", alpha=0.35)
            else:
                patch = Polygon(obstacle_corners(obs_norm), closed=True, color="gray", alpha=0.35)
            ax.add_patch(patch)

        cmap = plt.get_cmap("tab10")
        trails = []
        points = []
        for i in range(n):
            color = cmap(i % 10)
            (trail,) = ax.plot([], [], color=color, linewidth=1.8)
            (point,) = ax.plot([], [], marker="o", color=color, markersize=5)
            trails.append(trail)
            points.append(point)
            ax.scatter(goals[i, 0], goals[i, 1], color=color, marker="*", s=65)
        label_artist = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 2.0},
        )

        def _update(frame: int):
            artists = []
            for i in range(n):
                trails[i].set_data(pos[: frame + 1, i, 0], pos[: frame + 1, i, 1])
                points[i].set_data([pos[frame, i, 0]], [pos[frame, i, 1]])
                if unsafe is not None and unsafe.shape[0] > frame and unsafe[frame, i]:
                    points[i].set_marker("x")
                else:
                    points[i].set_marker("o")
                artists.extend([trails[i], points[i]])
            if frame_labels is not None:
                label_artist.set_text(frame_labels[frame])
            else:
                label_artist.set_text(f"t={frame}")
            artists.append(label_artist)
            return artists

        anim = FuncAnimation(fig, _update, frames=t, interval=max(20, int(1000 / max(1, fps))), blit=True)
        writer = PillowWriter(fps=fps)
        anim.save(str(output), writer=writer)
        plt.close(fig)
        return str(output)
