from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from hmarl_cbf.eval import EpisodeTrace, EvalEpisodeStats, TrajectoryRenderer, evaluate_summary
from hmarl_cbf.openrl_train.common import apply_config_section_defaults, build_core_env, load_yaml, save_json
from hmarl_cbf.openrl_train.pretrain_low_from_teacher import _build_low_env
from hmarl_cbf.openrl_train.teacher_dataset import (
    TeacherDatasetCollector,
    TeacherDatasetCollectorConfig,
    build_teacher_controller_from_config,
)


def _sample_points_from_perceived_obstacle(obs: Dict[str, Any]) -> List[np.ndarray]:
    item = dict(obs)
    obs_type = str(item.get("type", "")).strip().lower()
    if obs_type == "lidar_line":
        start = np.asarray(item.get("start", item.get("center", np.zeros(2, dtype=np.float32))), dtype=np.float32).reshape(2)
        end = np.asarray(item.get("end", item.get("center", np.zeros(2, dtype=np.float32))), dtype=np.float32).reshape(2)
        n_samples = 5
        return [
            ((1.0 - alpha) * start + alpha * end).astype(np.float32)
            for alpha in np.linspace(0.0, 1.0, num=n_samples, dtype=np.float32)
        ]
    center = np.asarray(item.get("center", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
    radius = float(item.get("radius", 0.0))
    if obs_type in {"circle", "lidar_circle"} and radius > 1e-6:
        return [center.astype(np.float32)] + [
            (center + radius * np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)).astype(np.float32)
            for theta in np.linspace(0.0, 2.0 * np.pi, num=8, endpoint=False, dtype=np.float32)
        ]
    return [center.astype(np.float32)]


def _frame_lidar_points_from_teacher_infos(low_infos: Dict[int, Dict[str, Any]]) -> np.ndarray:
    points: List[np.ndarray] = []
    for item in low_infos.values():
        perceived = list(item.get("perceived_obstacles", []))
        for obs in perceived:
            points.extend(_sample_points_from_perceived_obstacle(obs))
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    stacked = np.stack(points, axis=0).astype(np.float32, copy=False)
    return np.unique(np.round(stacked, decimals=6), axis=0).astype(np.float32, copy=False)


def _frame_lidar_obstacles_from_teacher_infos(low_infos: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    obstacles: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in low_infos.values():
        perceived = list(item.get("perceived_obstacles", []))
        for obs in perceived:
            obs_item = dict(obs)
            key = json.dumps(obs_item, sort_keys=True, default=lambda x: np.asarray(x).tolist())
            if key in seen:
                continue
            seen.add(key)
            obstacles.append(obs_item)
    return obstacles


def _summarize_lidar_obstacle_types(obstacles: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for obs in obstacles:
        key = str(obs.get("type", "unknown")).strip().lower()
        counts[key] = int(counts.get(key, 0) + 1)
    if not counts:
        return "lidar:none"
    parts = [f"{key}={counts[key]}" for key in sorted(counts.keys())]
    return "lidar:" + ",".join(parts)


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_output_dir(output_root: Path, run_name: str) -> Path:
    out_dir = output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _evaluate_episode(
    core,
    low_env,
    teacher_runner: TeacherDatasetCollector,
    *,
    episode_index: int,
    seed: int,
    capture_trace: bool,
) -> tuple[EvalEpisodeStats, Dict[str, Any], EpisodeTrace | None]:
    teacher_runner._active_episode_index = int(episode_index)
    teacher_runner._active_step_index = 0
    low_env.reset(seed=seed)
    if hasattr(teacher_runner.teacher_controller, "select_skill_ids"):
        initial_skill_map = teacher_runner.teacher_controller.select_skill_ids(low_env, low_env.possible_agents)
    else:
        initial_skill_map = {
            int(agent_id): teacher_runner._select_skill_for_agent(int(agent_id))
            for agent_id in low_env.possible_agents
        }
    low_env.set_skill_map(initial_skill_map, validate=True, require_all=True)

    agent_ids = list(core.agent_ids)
    n_agents = int(len(agent_ids))
    state0 = core.get_states()
    goals = np.stack([np.asarray(state0[aid].goal, dtype=np.float32).reshape(2) for aid in agent_ids], axis=0)
    obstacles = [dict(obs) for obs in core.get_obstacles()]

    trace_positions: List[np.ndarray] = []
    trace_unsafe: List[np.ndarray] = []
    frame_labels: List[str] = []
    trace_lidar_points: List[np.ndarray] = []
    trace_lidar_obstacles: List[List[Dict[str, Any]]] = []
    if capture_trace:
        trace_positions.append(
            np.stack([np.asarray(state0[aid].position, dtype=np.float32).reshape(2) for aid in agent_ids], axis=0)
        )
        trace_unsafe.append(np.zeros((n_agents,), dtype=bool))
        frame_labels.append("t=0")
        trace_lidar_points.append(np.zeros((0, 2), dtype=np.float32))
        trace_lidar_obstacles.append([])

    unsafe_any = {aid: False for aid in agent_ids}
    reached_any = {aid: False for aid in agent_ids}
    returns_by_agent = {aid: 0.0 for aid in agent_ids}
    physical_steps = 0
    qp_count = 0
    qp_feasible_count = 0
    qp_fallback_count = 0
    slack_values: List[float] = []
    cbf_slack_values: List[float] = []
    min_h_agent = float("inf")
    min_h_obstacle = float("inf")

    while True:
        teacher_runner._active_step_index = int(physical_steps)
        if low_env.has_pending_switch():
            pending_agents = list(low_env.pending_switch_agents)
            if hasattr(teacher_runner.teacher_controller, "select_skill_ids"):
                skill_map = teacher_runner.teacher_controller.select_skill_ids(low_env, pending_agents)
            else:
                skill_map = {
                    int(agent_id): teacher_runner._select_skill_for_agent(int(agent_id))
                    for agent_id in pending_agents
                }
            low_env.set_skill_map(skill_map, validate=True, require_all=False)

        phi_zeros = {
            int(agent_id): np.zeros((int(low_env.phi_dim),), dtype=np.float32)
            for agent_id in low_env.agents
        }
        _, _, _, _, low_infos = low_env.step(phi_zeros)
        info = dict(core.last_info)
        safety_metrics = dict(info.get("safety_metrics", {}))
        unsafe_flags = dict(info.get("unsafe_flags", {}))
        reach_flags = dict(info.get("reach_flags", {}))

        physical_steps += 1
        for aid in agent_ids:
            unsafe_any[aid] = bool(unsafe_any[aid] or unsafe_flags.get(aid, False))
            reached_any[aid] = bool(reached_any[aid] or reach_flags.get(aid, False))
            returns_by_agent[aid] += float(core.last_rewards.get(aid, 0.0))
            metric = safety_metrics.get(aid)
            if metric is not None:
                min_h_agent = min(min_h_agent, float(metric.min_h_agent))
                min_h_obstacle = min(min_h_obstacle, float(metric.min_h_obstacle))

            solution_info = dict(low_infos.get(aid, {}))
            if "teacher_qp_feasible" in solution_info:
                qp_count += 1
                qp_feasible_count += int(bool(solution_info.get("teacher_qp_feasible", False)))
                qp_fallback_count += int(bool(solution_info.get("teacher_qp_used_fallback", False)))
                slack = np.asarray(solution_info.get("teacher_qp_slack", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
                cbf_slack = np.asarray(
                    solution_info.get("teacher_qp_cbf_slack", np.zeros((0,), dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
                if slack.size > 0:
                    slack_values.extend(float(v) for v in slack.tolist())
                if cbf_slack.size > 0:
                    cbf_slack_values.extend(float(v) for v in cbf_slack.tolist())

        if capture_trace:
            frame_lidar_obstacles = _frame_lidar_obstacles_from_teacher_infos(low_infos)
            next_states = core.get_states()
            trace_positions.append(
                np.stack([np.asarray(next_states[aid].position, dtype=np.float32).reshape(2) for aid in agent_ids], axis=0)
            )
            trace_unsafe.append(np.asarray([bool(unsafe_flags.get(aid, False)) for aid in agent_ids], dtype=bool))
            wind = np.asarray(info.get("wind_accel", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
            frame_labels.append(
                f"t={int(info.get('step_count', physical_steps))} "
                f"wind=({wind[0]:+0.2f},{wind[1]:+0.2f}) "
                f"{_summarize_lidar_obstacle_types(frame_lidar_obstacles)}"
            )
            trace_lidar_points.append(_frame_lidar_points_from_teacher_infos(low_infos))
            trace_lidar_obstacles.append(frame_lidar_obstacles)

        if bool(core.last_terminated or core.last_truncated or not low_env.agents):
            break

    safe_reach_count = int(sum(1 for aid in agent_ids if reached_any[aid] and not unsafe_any[aid]))
    safe_reach_ratio = float(safe_reach_count / max(1, n_agents))
    success_round = bool(safe_reach_count == n_agents)
    reach_rate = float(sum(1.0 for aid in agent_ids if reached_any[aid]) / max(1, n_agents))
    collision_rate = float(sum(1.0 for aid in agent_ids if unsafe_any[aid]) / max(1, n_agents))
    qp_feasible_rate = float(qp_feasible_count / max(1, qp_count))
    qp_fallback_rate = float(qp_fallback_count / max(1, qp_count))
    avg_return = float(np.mean(list(returns_by_agent.values()))) if returns_by_agent else 0.0
    slack_mean = float(np.mean(slack_values)) if slack_values else 0.0
    slack_max = float(np.max(slack_values)) if slack_values else 0.0
    cbf_slack_mean = float(np.mean(cbf_slack_values)) if cbf_slack_values else 0.0
    cbf_slack_max = float(np.max(cbf_slack_values)) if cbf_slack_values else 0.0

    if min_h_agent == float("inf"):
        min_h_agent = 0.0
    if min_h_obstacle == float("inf"):
        min_h_obstacle = 0.0

    ep_stats = EvalEpisodeStats(
        episode_index=int(episode_index),
        success=success_round,
        reach_rate=reach_rate,
        collision_rate=collision_rate,
        min_h_agent=float(min_h_agent),
        min_h_obstacle=float(min_h_obstacle),
        avg_traj_length=float(physical_steps),
        skill_switches=0,
        qp_feasible_rate=qp_feasible_rate,
        steps=int(physical_steps),
        episode_return_mean=avg_return,
        safe_reach_ratio=safe_reach_ratio,
        extras={
            "safe_reach_count": int(safe_reach_count),
            "n_agents": int(n_agents),
            "slack_mean": slack_mean,
            "slack_max": slack_max,
            "cbf_slack_mean": cbf_slack_mean,
            "cbf_slack_max": cbf_slack_max,
            "qp_count": int(qp_count),
            "qp_fallback_count": int(qp_fallback_count),
            "qp_fallback_rate": qp_fallback_rate,
        },
    )
    row = {
        "episode": int(episode_index),
        "steps": int(physical_steps),
        "safe_reach_count": int(safe_reach_count),
        "n_agents": int(n_agents),
        "safe_reach_ratio": safe_reach_ratio,
        "success_round": int(success_round),
        "reach_rate": reach_rate,
        "collision_rate": collision_rate,
        "qp_feasible_rate": qp_feasible_rate,
        "qp_fallback_count": int(qp_fallback_count),
        "qp_fallback_rate": qp_fallback_rate,
        "mean_return": avg_return,
        "slack_mean": slack_mean,
        "slack_max": slack_max,
        "cbf_slack_mean": cbf_slack_mean,
        "cbf_slack_max": cbf_slack_max,
        "skill_switches": 0,
        "min_h_agent": float(min_h_agent),
        "min_h_obstacle": float(min_h_obstacle),
    }
    trace = None
    if capture_trace and trace_positions:
        trace = EpisodeTrace(
            positions=np.stack(trace_positions, axis=0),
            goals=np.asarray(goals, dtype=np.float32),
            obstacles=obstacles,
            unsafe_flags=np.stack(trace_unsafe, axis=0),
            frame_labels=list(frame_labels),
            lidar_points=list(trace_lidar_points),
            lidar_obstacles=list(trace_lidar_obstacles),
        )
    return ep_stats, row, trace


def evaluate_teacher(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    core = build_core_env(cfg)
    low_env = _build_low_env(cfg, core, torch_device="cpu")
    teacher = build_teacher_controller_from_config(
        cfg,
        teacher_source=str(args.teacher_source),
        teacher_checkpoint=str(args.teacher_checkpoint),
    )
    teacher_runner = TeacherDatasetCollector(
        low_env,
        teacher,
        TeacherDatasetCollectorConfig(
            episodes=1,
            max_steps_per_episode=0,
            seed=int(args.seed),
            skill_selection_mode=str(args.teacher_skill_selection_mode),
        ),
    )
    output_dir = _make_output_dir(Path(args.output_root), args.run_name)
    renderer = TrajectoryRenderer(world_size=float(cfg["env"]["world_size"]))
    render_media = bool(args.render_gif or args.render_png)
    render_gif = bool(args.render_gif) or not bool(args.render_png)
    media_dir = output_dir / "media"
    if render_media:
        media_dir.mkdir(parents=True, exist_ok=True)

    episode_stats: List[EvalEpisodeStats] = []
    rows: List[Dict[str, Any]] = []
    total_safe_reach = 0
    total_agents = 0
    success_rounds = 0
    relax_values: List[float] = []
    cbf_relax_values: List[float] = []

    for ep in range(int(args.episodes)):
        stats, row, trace = _evaluate_episode(
            core,
            low_env,
            teacher_runner,
            episode_index=ep,
            seed=int(args.seed + ep),
            capture_trace=bool(render_media and ep < int(args.render_episodes)),
        )
        media_path = ""
        if trace is not None:
            suffix = "gif" if render_gif else "png"
            media_path = str(media_dir / f"episode_{ep:03d}.{suffix}")
            if render_gif:
                renderer.render_gif(trace, media_path, fps=int(args.fps))
            else:
                renderer.render_static(trace, media_path)
        row["media_path"] = media_path
        rows.append(row)
        episode_stats.append(stats)
        total_safe_reach += int(stats.extras["safe_reach_count"])
        total_agents += int(stats.extras["n_agents"])
        success_rounds += int(stats.success)
        relax_values.append(float(stats.extras["slack_mean"]))
        cbf_relax_values.append(float(stats.extras["cbf_slack_mean"]))
        print(
            f"[episode {ep + 1}/{int(args.episodes)}] "
            f"safe_reach={int(stats.extras['safe_reach_count'])}/{int(stats.extras['n_agents'])} "
            f"success={int(stats.success)} "
            f"ret={float(stats.episode_return_mean):.4f} "
            f"steps={int(stats.steps)}"
        )

    summary_stats = evaluate_summary(episode_stats)
    teacher_cfg = dict(cfg.get("teacher_baseline", {}))
    summary = {
        "episodes": int(args.episodes),
        "teacher_source": str(args.teacher_source),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "nominal_mode": str(teacher_cfg.get("nominal_mode", "lqr")),
        "total_safe_reach": int(total_safe_reach),
        "total_agents": int(total_agents),
        "overall_safe_reach_ratio": float(total_safe_reach / max(1, total_agents)),
        "success_rounds": int(success_rounds),
        "success_round_rate": float(success_rounds / max(1, int(args.episodes))),
        "mean_return": float(summary_stats.avg_episode_return),
        "collision_rate": float(summary_stats.collision_rate),
        "reach_rate": float(summary_stats.reach_rate),
        "avg_steps": float(summary_stats.avg_steps),
        "avg_skill_switches": 0.0,
        "qp_feasible_rate": float(summary_stats.qp_feasible_rate),
        "min_h_agent": float(summary_stats.min_h_agent),
        "min_h_obstacle": float(summary_stats.min_h_obstacle),
        "slack_mean": float(np.mean(relax_values)) if relax_values else 0.0,
        "slack_max": float(max((row["slack_max"] for row in rows), default=0.0)),
        "cbf_slack_mean": float(np.mean(cbf_relax_values)) if cbf_relax_values else 0.0,
        "cbf_slack_max": float(max((row["cbf_slack_max"] for row in rows), default=0.0)),
        "relax_mean": float(np.mean(relax_values)) if relax_values else 0.0,
        "relax_max": float(max((row["slack_max"] for row in rows), default=0.0)),
        "output_dir": str(output_dir),
    }

    _write_rows_csv(output_dir / "episode_results.csv", rows)
    save_json(output_dir / "summary.json", summary)

    print("==== Teacher Evaluation Summary ====")
    print(f"safe_reach_total: {total_safe_reach}/{total_agents} ({summary['overall_safe_reach_ratio']:.4f})")
    print(f"success_rounds: {success_rounds}/{int(args.episodes)} ({summary['success_round_rate']:.4f})")
    print(f"mean_return: {summary['mean_return']:.4f}")
    print(f"relax_mean: {summary['relax_mean']:.6f}")
    print(f"relax_max: {summary['relax_max']:.6f}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent evaluation for the configured teacher controller")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--teacher-source", type=str, default=None, choices=["baseline", "model_checkpoint"])
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--teacher-skill-selection-mode", type=str, default=None, choices=["cyclic", "random"])
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--render-gif", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--render-png", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--render-episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    args = apply_config_section_defaults(
        args,
        cfg,
        section="openrl_teacher_eval",
        defaults={
            "episodes": 10,
            "seed": 0,
            "teacher_source": "model_checkpoint",
            "teacher_checkpoint": "artifacts/hmarl_cbf_paper_async/phi_ppo_async_trapaware_stuck_rcbf_lidarpointcbf_topk_hardresume_round2/checkpoints/last.pt",
            "teacher_skill_selection_mode": "cyclic",
            "output_root": "artifacts/openrl_eval",
            "run_name": "eval_teacher_baseline",
            "render_gif": False,
            "render_png": False,
            "render_episodes": 1,
            "fps": 8,
            "summary_json": "",
        },
    )
    summary = evaluate_teacher(cfg, args)
    if args.summary_json:
        save_json(args.summary_json, summary)


if __name__ == "__main__":
    main()
