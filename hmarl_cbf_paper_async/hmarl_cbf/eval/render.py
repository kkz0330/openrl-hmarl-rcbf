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
    lidar_points: List[np.ndarray] | None = None  # per-frame selected lidar points, each [M, 2]
    lidar_obstacles: List[List[Dict[str, np.ndarray | float]]] | None = None  # per-frame fitted lidar obstacles


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
        if trace.lidar_points:
            last_points = np.asarray(trace.lidar_points[-1], dtype=np.float32).reshape(-1, 2)
            if last_points.size > 0:
                ax.scatter(last_points[:, 0], last_points[:, 1], color="darkorange", marker=".", s=28, alpha=0.9)
        if trace.lidar_obstacles:
            self._draw_lidar_geometry_overlay(ax, trace.lidar_obstacles[-1])

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
        lidar_points = list(trace.lidar_points) if trace.lidar_points is not None else None
        lidar_obstacles = list(trace.lidar_obstacles) if trace.lidar_obstacles is not None else None
        t, n, _ = pos.shape
        assert goals.shape == (n, 2)
        if frame_labels is not None and len(frame_labels) < t:
            frame_labels = frame_labels + [""] * (t - len(frame_labels))
        if lidar_points is not None and len(lidar_points) < t:
            lidar_points = lidar_points + [np.zeros((0, 2), dtype=np.float32) for _ in range(t - len(lidar_points))]
        if lidar_obstacles is not None and len(lidar_obstacles) < t:
            lidar_obstacles = lidar_obstacles + [[] for _ in range(t - len(lidar_obstacles))]

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
        lidar_artist = ax.scatter([], [], color="darkorange", marker=".", s=28, alpha=0.9)
        max_line_obs = 0
        max_circle_obs = 0
        if lidar_obstacles is not None:
            for frame_obs in lidar_obstacles:
                line_count = sum(1 for obs in frame_obs if str(obs.get("type", "")).strip().lower() == "lidar_line")
                circle_count = sum(1 for obs in frame_obs if str(obs.get("type", "")).strip().lower() == "lidar_circle")
                max_line_obs = max(max_line_obs, int(line_count))
                max_circle_obs = max(max_circle_obs, int(circle_count))
        lidar_line_artists = []
        for _ in range(max_line_obs):
            (artist,) = ax.plot([], [], color="darkorange", linestyle="--", linewidth=1.6, alpha=0.95)
            artist.set_visible(False)
            lidar_line_artists.append(artist)
        lidar_circle_artists = []
        for _ in range(max_circle_obs):
            patch = plt.Circle((0.0, 0.0), 1.0, fill=False, edgecolor="darkorange", linestyle="--", linewidth=1.6, alpha=0.95)
            patch.set_visible(False)
            ax.add_patch(patch)
            lidar_circle_artists.append(patch)

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
            if lidar_points is not None:
                pts = np.asarray(lidar_points[frame], dtype=np.float32).reshape(-1, 2)
                if pts.size > 0:
                    lidar_artist.set_offsets(pts)
                else:
                    lidar_artist.set_offsets(np.zeros((0, 2), dtype=np.float32))
            if lidar_obstacles is not None:
                self._update_lidar_geometry_artists(
                    lidar_obstacles[frame],
                    lidar_line_artists,
                    lidar_circle_artists,
                )
            artists.extend([label_artist, lidar_artist, *lidar_line_artists, *lidar_circle_artists])
            return artists

        anim = FuncAnimation(fig, _update, frames=t, interval=max(20, int(1000 / max(1, fps))), blit=True)
        writer = PillowWriter(fps=fps)
        anim.save(str(output), writer=writer)
        plt.close(fig)
        return str(output)

    @staticmethod
    def _draw_lidar_geometry_overlay(ax, obstacles: List[Dict[str, np.ndarray | float]]) -> None:
        for obs in obstacles:
            obs_type = str(obs.get("type", "")).strip().lower()
            if obs_type == "lidar_line":
                start = np.asarray(obs.get("start", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
                end = np.asarray(obs.get("end", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
                ax.plot(
                    [float(start[0]), float(end[0])],
                    [float(start[1]), float(end[1])],
                    color="darkorange",
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.95,
                )
            elif obs_type == "lidar_circle":
                center = np.asarray(obs.get("center", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
                radius = float(obs.get("radius", 0.0))
                if radius > 1e-6:
                    ax.add_patch(
                        plt.Circle(
                            (float(center[0]), float(center[1])),
                            radius,
                            fill=False,
                            edgecolor="darkorange",
                            linestyle="--",
                            linewidth=1.6,
                            alpha=0.95,
                        )
                    )

    @staticmethod
    def _update_lidar_geometry_artists(
        obstacles: List[Dict[str, np.ndarray | float]],
        line_artists,
        circle_artists,
    ) -> None:
        line_items = [obs for obs in obstacles if str(obs.get("type", "")).strip().lower() == "lidar_line"]
        circle_items = [obs for obs in obstacles if str(obs.get("type", "")).strip().lower() == "lidar_circle"]

        for idx, artist in enumerate(line_artists):
            if idx < len(line_items):
                obs = line_items[idx]
                start = np.asarray(obs.get("start", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
                end = np.asarray(obs.get("end", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
                artist.set_data(
                    [float(start[0]), float(end[0])],
                    [float(start[1]), float(end[1])],
                )
                artist.set_visible(True)
            else:
                artist.set_data([], [])
                artist.set_visible(False)

        for idx, patch in enumerate(circle_artists):
            if idx < len(circle_items):
                obs = circle_items[idx]
                center = np.asarray(obs.get("center", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
                radius = float(obs.get("radius", 0.0))
                patch.center = (float(center[0]), float(center[1]))
                patch.radius = max(0.0, radius)
                patch.set_visible(radius > 1e-6)
            else:
                patch.set_visible(False)
