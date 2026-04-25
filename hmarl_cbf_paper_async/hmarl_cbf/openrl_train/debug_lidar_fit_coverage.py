from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.env.obstacles import copy_obstacle, fit_lidar_local_obstacles, normalize_obstacle, obstacle_corners, obstacle_surface_distance
from hmarl_cbf.openrl_train.common import load_yaml
from hmarl_cbf.types import LIDAR_HIT_OBSTACLE, LidarScan

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
except ImportError:  # pragma: no cover - optional backend
    plt = None  # type: ignore[assignment]
    Polygon = None  # type: ignore[assignment]


def _wrap_segments(indices: np.ndarray, n_total: int) -> List[np.ndarray]:
    if indices.size == 0:
        return []
    segments: List[List[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        if int(idx) == segments[-1][-1] + 1:
            segments[-1].append(int(idx))
        else:
            segments.append([int(idx)])
    if len(segments) > 1 and segments[0][0] == 0 and segments[-1][-1] == n_total - 1:
        merged = segments[-1] + segments[0]
        segments = [merged] + segments[1:-1]
    return [np.asarray(seg, dtype=np.int32) for seg in segments]


def _scan_visible_points(scan: LidarScan, interpolation_points_per_gap: int) -> tuple[np.ndarray, np.ndarray]:
    hit_points = np.asarray(scan.hit_points, dtype=np.float32).reshape(-1, 2)
    hit_valid = np.asarray(scan.hit_valid, dtype=np.bool_).reshape(-1)
    hit_kinds = np.asarray(scan.hit_kinds, dtype=np.int32).reshape(-1)
    valid_indices = np.flatnonzero(hit_valid & (hit_kinds == int(LIDAR_HIT_OBSTACLE)))
    hit_samples = hit_points[valid_indices].astype(np.float32) if valid_indices.size > 0 else np.zeros((0, 2), dtype=np.float32)

    gap_points: List[np.ndarray] = []
    if interpolation_points_per_gap > 0:
        for seg in _wrap_segments(valid_indices, n_total=int(hit_points.shape[0])):
            if seg.shape[0] < 2:
                continue
            for idx in range(int(seg.shape[0]) - 1):
                p0 = hit_points[int(seg[idx])]
                p1 = hit_points[int(seg[idx + 1])]
                for alpha_idx in range(1, interpolation_points_per_gap + 1):
                    alpha = float(alpha_idx / (interpolation_points_per_gap + 1))
                    gap_points.append(((1.0 - alpha) * p0 + alpha * p1).astype(np.float32))
    interp_samples = (
        np.stack(gap_points, axis=0).astype(np.float32)
        if gap_points
        else np.zeros((0, 2), dtype=np.float32)
    )
    return hit_samples, interp_samples


def _coverage_stats(points: np.ndarray, fitted_obstacles: List[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    if points.size == 0:
        return {
            "count": 0,
            "covered_count": 0,
            "coverage_rate": 1.0,
            "worst_clearance": 0.0,
            "worst_point": None,
        }

    worst_clearance = -float("inf")
    worst_point: List[float] | None = None
    covered_count = 0

    for point in np.asarray(points, dtype=np.float32).reshape(-1, 2):
        if fitted_obstacles:
            best = min(float(obstacle_surface_distance(point, obs)) for obs in fitted_obstacles)
        else:
            best = float("inf")
        if best <= tolerance:
            covered_count += 1
        if best > worst_clearance:
            worst_clearance = best
            worst_point = np.asarray(point, dtype=np.float32).reshape(2).tolist()

    return {
        "count": int(points.shape[0]),
        "covered_count": int(covered_count),
        "coverage_rate": float(covered_count / max(int(points.shape[0]), 1)),
        "worst_clearance": float(worst_clearance),
        "worst_point": worst_point,
    }


def _fit_type_counts(obstacles: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for obs in obstacles:
        obs_type = str(obs.get("type", "unknown"))
        counts[obs_type] = int(counts.get(obs_type, 0) + 1)
    return counts


def _draw_obstacle(ax, obstacle: Dict[str, Any], *, facecolor: str, edgecolor: str, alpha: float, linestyle: str, linewidth: float) -> None:
    obs = normalize_obstacle(obstacle)
    center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
    if obs["type"] in ("circle", "point", "lidar_point", "lidar_circle"):
        radius = float(obs.get("radius", 0.0))
        patch = plt.Circle(
            (float(center[0]), float(center[1])),
            radius,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        ax.add_patch(patch)
        return
    if obs["type"] == "lidar_line":
        start = np.asarray(obs["start"], dtype=np.float32).reshape(2)
        end = np.asarray(obs["end"], dtype=np.float32).reshape(2)
        ax.plot(
            [float(start[0]), float(end[0])],
            [float(start[1]), float(end[1])],
            color=edgecolor,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=max(alpha, 0.85),
        )
        return
    patch = Polygon(
        obstacle_corners(obs),
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    ax.add_patch(patch)


def _render_case(
    *,
    output_path: Path,
    world_size: float,
    origin: np.ndarray,
    true_obstacles: List[Dict[str, Any]],
    fitted_obstacles: List[Dict[str, Any]],
    hit_points: np.ndarray,
    interp_points: np.ndarray,
    row: Dict[str, Any],
) -> str:
    if plt is None or Polygon is None:
        raise RuntimeError("matplotlib is required for lidar fit visualization")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
    ax.set_aspect("equal")

    for obstacle in true_obstacles:
        _draw_obstacle(
            ax,
            obstacle,
            facecolor="lightgray",
            edgecolor="dimgray",
            alpha=0.35,
            linestyle="-",
            linewidth=1.2,
        )
    for obstacle in fitted_obstacles:
        _draw_obstacle(
            ax,
            obstacle,
            facecolor="none",
            edgecolor="tab:blue",
            alpha=1.0,
            linestyle="--",
            linewidth=2.0,
        )

    hit_points = np.asarray(hit_points, dtype=np.float32).reshape(-1, 2)
    interp_points = np.asarray(interp_points, dtype=np.float32).reshape(-1, 2)
    if hit_points.size > 0:
        ax.scatter(hit_points[:, 0], hit_points[:, 1], s=24, c="tab:red", marker="o", label="LiDAR hit points")
    if interp_points.size > 0:
        ax.scatter(interp_points[:, 0], interp_points[:, 1], s=12, c="darkorange", marker=".", alpha=0.9, label="Interpolated visible surface")

    origin = np.asarray(origin, dtype=np.float32).reshape(2)
    ax.scatter([float(origin[0])], [float(origin[1])], s=55, c="black", marker="x", label="Agent / LiDAR origin")
    ax.set_title(
        "LiDAR Fit Coverage\n"
        f"ep={int(row['episode'])} seed={int(row['seed'])} agent={int(row['agent_id'])} | "
        f"hit={float(row['hit_point_stats']['coverage_rate']):.3f} "
        f"interp={float(row['interp_point_stats']['coverage_rate']):.3f}"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def _evaluate_agent(
    *,
    episode_index: int,
    seed: int,
    agent_id: int,
    scan: LidarScan,
    fit_kwargs: Dict[str, Any],
    interpolation_points_per_gap: int,
    coverage_tolerance: float,
    true_obstacles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fitted = [copy_obstacle(obs) for obs in fit_lidar_local_obstacles(scan, **fit_kwargs)]
    hit_points, interp_points = _scan_visible_points(scan, interpolation_points_per_gap)
    hit_stats = _coverage_stats(hit_points, fitted, coverage_tolerance)
    interp_stats = _coverage_stats(interp_points, fitted, coverage_tolerance)
    return {
        "episode": int(episode_index),
        "seed": int(seed),
        "agent_id": int(agent_id),
        "origin": np.asarray(scan.origin, dtype=np.float32).reshape(2).tolist(),
        "n_hit_points": int(hit_points.shape[0]),
        "n_interp_points": int(interp_points.shape[0]),
        "n_fitted_obstacles": int(len(fitted)),
        "fitted_type_counts": _fit_type_counts(fitted),
        "hit_point_stats": hit_stats,
        "interp_point_stats": interp_stats,
        "true_obstacles": [
            {
                k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                for k, v in copy_obstacle(obs).items()
            }
            for obs in true_obstacles
        ],
        "hit_points": np.asarray(hit_points, dtype=np.float32).reshape(-1, 2).tolist(),
        "interp_points": np.asarray(interp_points, dtype=np.float32).reshape(-1, 2).tolist(),
        "fitted_obstacles": [
            {
                k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                for k, v in obs.items()
            }
            for obs in fitted
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone LiDAR obstacle-fitting coverage debug script.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=0.05,
        help="A visible point counts as covered if its distance to any fitted obstacle is <= this tolerance.",
    )
    parser.add_argument(
        "--interpolation-points-per-gap",
        type=int,
        default=3,
        help="How many interpolated visible-surface points to add between adjacent obstacle hit points in one LiDAR segment.",
    )
    parser.add_argument("--min-segment-points", type=int, default=None)
    parser.add_argument("--line-fit-max-residual", type=float, default=None)
    parser.add_argument("--circle-fit-max-residual", type=float, default=None)
    parser.add_argument("--circle-radius-min", type=float, default=None)
    parser.add_argument("--circle-radius-max", type=float, default=None)
    parser.add_argument("--max-print", type=int, default=10)
    parser.add_argument("--render-dir", type=str, default="")
    parser.add_argument("--render-max-cases", type=int, default=12)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    env_cfg = dict(cfg.get("env", {}))
    env = MultiUAV2DEnv(**env_cfg)

    fit_kwargs = {
        "max_range": float(env.lidar.max_range),
        "min_segment_points": int(
            args.min_segment_points
            if args.min_segment_points is not None
            else env_cfg.get("lidar_cbf_min_segment_points", 2)
        ),
        "line_fit_max_residual": float(
            args.line_fit_max_residual
            if args.line_fit_max_residual is not None
            else env_cfg.get("lidar_cbf_line_fit_max_residual", 0.08)
        ),
        "circle_fit_max_residual": float(
            args.circle_fit_max_residual
            if args.circle_fit_max_residual is not None
            else env_cfg.get("lidar_cbf_circle_fit_max_residual", 0.08)
        ),
        "circle_radius_min": float(
            args.circle_radius_min
            if args.circle_radius_min is not None
            else env_cfg.get("lidar_cbf_circle_radius_min", 0.05)
        ),
        "circle_radius_max": float(
            args.circle_radius_max
            if args.circle_radius_max is not None
            else env_cfg.get("lidar_cbf_circle_radius_max", 100.0)
        ),
    }

    rows: List[Dict[str, Any]] = []
    printed = 0
    for ep in range(int(args.episodes)):
        seed = int(args.seed) + ep
        obs, _ = env.reset(seed=seed)
        true_obstacles = [copy_obstacle(obs_item) for obs_item in env.get_obstacles()]
        for agent_id, agent_obs in obs.items():
            row = _evaluate_agent(
                episode_index=ep,
                seed=seed,
                agent_id=int(agent_id),
                scan=agent_obs["low"].lidar_scan,
                fit_kwargs=fit_kwargs,
                interpolation_points_per_gap=int(args.interpolation_points_per_gap),
                coverage_tolerance=float(args.coverage_tolerance),
                true_obstacles=true_obstacles,
            )
            if args.render_dir and len(rows) < int(args.render_max_cases):
                render_path = Path(args.render_dir) / (
                    f"ep_{int(row['episode']):03d}_seed_{int(row['seed'])}_agent_{int(row['agent_id'])}.png"
                )
                row["render_path"] = _render_case(
                    output_path=render_path,
                    world_size=float(env.world_size),
                    origin=np.asarray(row["origin"], dtype=np.float32),
                    true_obstacles=[dict(item) for item in row["true_obstacles"]],
                    fitted_obstacles=[dict(item) for item in row["fitted_obstacles"]],
                    hit_points=np.asarray(row["hit_points"], dtype=np.float32),
                    interp_points=np.asarray(row["interp_points"], dtype=np.float32),
                    row=row,
                )
            rows.append(row)
            if printed < int(args.max_print):
                print(
                    "lidar_fit_debug_row",
                    json.dumps(
                        {
                            "episode": int(row["episode"]),
                            "seed": int(row["seed"]),
                            "agent_id": int(row["agent_id"]),
                            "n_fitted_obstacles": int(row["n_fitted_obstacles"]),
                            "fitted_type_counts": row["fitted_type_counts"],
                            "hit_point_coverage_rate": float(row["hit_point_stats"]["coverage_rate"]),
                            "interp_point_coverage_rate": float(row["interp_point_stats"]["coverage_rate"]),
                            "hit_worst_clearance": float(row["hit_point_stats"]["worst_clearance"]),
                            "interp_worst_clearance": float(row["interp_point_stats"]["worst_clearance"]),
                            "render_path": row.get("render_path", ""),
                        },
                        ensure_ascii=False,
                    ),
                )
                printed += 1

    hit_total = sum(int(row["hit_point_stats"]["count"]) for row in rows)
    hit_covered = sum(int(row["hit_point_stats"]["covered_count"]) for row in rows)
    interp_total = sum(int(row["interp_point_stats"]["count"]) for row in rows)
    interp_covered = sum(int(row["interp_point_stats"]["covered_count"]) for row in rows)
    worst_hit = max(rows, key=lambda row: float(row["hit_point_stats"]["worst_clearance"])) if rows else None
    worst_interp = max(rows, key=lambda row: float(row["interp_point_stats"]["worst_clearance"])) if rows else None

    summary = {
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "coverage_tolerance": float(args.coverage_tolerance),
        "interpolation_points_per_gap": int(args.interpolation_points_per_gap),
        "fit_kwargs": fit_kwargs,
        "n_rows": int(len(rows)),
        "hit_point_coverage_rate": float(hit_covered / max(hit_total, 1)),
        "interp_point_coverage_rate": float(interp_covered / max(interp_total, 1)),
        "hit_point_total": int(hit_total),
        "interp_point_total": int(interp_total),
        "worst_hit_clearance": float(worst_hit["hit_point_stats"]["worst_clearance"]) if worst_hit is not None else 0.0,
        "worst_interp_clearance": float(worst_interp["interp_point_stats"]["worst_clearance"]) if worst_interp is not None else 0.0,
        "worst_hit_case": worst_hit,
        "worst_interp_case": worst_interp,
        "rows": rows,
    }

    print(
        "lidar_fit_debug_summary",
        json.dumps(
            {
                "episodes": int(summary["episodes"]),
                "n_rows": int(summary["n_rows"]),
                "hit_point_coverage_rate": float(summary["hit_point_coverage_rate"]),
                "interp_point_coverage_rate": float(summary["interp_point_coverage_rate"]),
                "worst_hit_clearance": float(summary["worst_hit_clearance"]),
                "worst_interp_clearance": float(summary["worst_interp_clearance"]),
            },
            ensure_ascii=False,
        ),
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
