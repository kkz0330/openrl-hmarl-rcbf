from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to load config") from exc

from hmarl_cbf.baselines import GCBFStyleHandcraftedConfig, GCBFStyleHandcraftedController
from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.eval import EpisodeTrace, TrajectoryRenderer
from hmarl_cbf.scenarios import build_fixed_scene, fixed_scene_names


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GCBF-style handcrafted CBF-QP baseline directly on the HMARL environment."
    )
    parser.add_argument("--config", type=str, default="configs/hmarl_cbf/default_async_onpolicy_gcbfplus_trapaware_mixedrect.yaml")
    parser.add_argument("--output-root", type=str, default="artifacts/hmarl_cbf_baseline")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--render-gif", action="store_true")
    parser.add_argument("--render-png", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument(
        "--scenario",
        type=str,
        default="",
        choices=[""] + fixed_scene_names(),
        help="Optional built-in fixed scenario.",
    )
    parser.add_argument("--states-json", type=str, default="")
    parser.add_argument("--obstacles-json", type=str, default="")
    return parser.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _make_run_dir(output_root: Path, run_name: str) -> Path:
    if run_name:
        out = output_root / run_name
    else:
        out = output_root / time.strftime("gcbf_style_baseline_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "media").mkdir(parents=True, exist_ok=True)
    return out


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _render_episode(
    renderer: TrajectoryRenderer,
    positions: List[np.ndarray],
    unsafe_flags: List[np.ndarray],
    frame_labels: List[str],
    goals: np.ndarray,
    obstacles: List[Dict[str, np.ndarray | float]],
    out_path: Path,
    render_gif: bool,
    fps: int,
) -> str:
    trace = EpisodeTrace(
        positions=np.stack(positions, axis=0),
        goals=goals,
        obstacles=obstacles,
        unsafe_flags=np.stack(unsafe_flags, axis=0),
        frame_labels=frame_labels,
    )
    if render_gif:
        return renderer.render_gif(trace, out_path, fps=max(1, int(fps)))
    return renderer.render_static(trace, out_path)


def _parse_fixed_scene(args: argparse.Namespace, env_cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    if args.states_json:
        states = json.loads(args.states_json)
        obstacles = json.loads(args.obstacles_json) if args.obstacles_json else []
        return {"states": states, "obstacles": obstacles}
    if args.scenario:
        states, obstacles = build_fixed_scene(
            args.scenario,
            world_size=float(env_cfg["world_size"]),
            agent_radius=float(env_cfg["agent_radius"]),
        )
        return {"states": states, "obstacles": obstacles}
    return None


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    seed = int(cfg.get("seed", 42) if int(args.seed) < 0 else int(args.seed))
    _seed_all(seed)

    env_cfg = dict(cfg["env"])
    env = MultiUAV2DEnv(**env_cfg)

    style_cfg = GCBFStyleHandcraftedConfig.from_mapping(
        {
            **cfg.get("gcbf_style_handcrafted_baseline", {}),
            "action_limit": env_cfg["action_limit"],
            "velocity_limit": env_cfg["velocity_limit"],
            "car_radius": env_cfg["agent_radius"],
            "n_rays": env_cfg["lidar_beams"],
            "dt": env_cfg["dt"],
            "comm_radius": cfg.get("gcbf_style_handcrafted_baseline", {}).get(
                "comm_radius",
                max(float(env_cfg.get("lidar_range", 3.0)), float(env_cfg.get("neighbor_radius", 3.0))),
            ),
        }
    )
    baseline = GCBFStyleHandcraftedController(style_cfg)

    run_dir = _make_run_dir(Path(args.output_root), args.run_name)
    snapshot = dict(cfg)
    snapshot["gcbf_style_handcrafted_baseline"] = {
        "alpha": float(style_cfg.alpha),
        "k": int(style_cfg.k),
        "action_limit": float(style_cfg.action_limit),
        "velocity_limit": float(style_cfg.velocity_limit),
        "comm_radius": float(style_cfg.comm_radius),
        "car_radius": float(style_cfg.car_radius),
        "n_rays": int(style_cfg.n_rays),
        "mass": float(style_cfg.mass),
        "dt": float(style_cfg.dt),
        "q_pos": float(style_cfg.q_pos),
        "q_vel": float(style_cfg.q_vel),
        "r_input": float(style_cfg.r_input),
        "relax_penalty": float(style_cfg.relax_penalty),
    }
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    renderer = TrajectoryRenderer(world_size=float(env_cfg["world_size"]))

    fixed_scene = _parse_fixed_scene(args, env_cfg)
    rows: List[Dict[str, Any]] = []
    total_safe_reach = 0
    total_agents = 0
    success_rounds = 0
    mean_return_sum = 0.0
    relax_sum = 0.0
    relax_max = 0.0
    relax_cnt = 0

    render_enabled = bool(args.render_gif) or bool(args.render_png)
    render_gif = bool(args.render_gif)
    n_episodes = max(1, int(args.episodes))
    for ep in range(n_episodes):
        if fixed_scene is not None:
            obs, _ = env.reset(seed=seed + ep, states=fixed_scene["states"], obstacles=fixed_scene["obstacles"])
        else:
            obs, _ = env.reset(seed=seed + ep)
        agent_ids = sorted(obs.keys())
        reached_any = {aid: False for aid in agent_ids}
        unsafe_any = {aid: False for aid in agent_ids}
        ep_return = {aid: 0.0 for aid in agent_ids}
        positions: List[np.ndarray] = []
        unsafe_trace: List[np.ndarray] = []
        frame_labels: List[str] = []
        goals = np.stack([s.goal for s in env.get_agent_states()], axis=0).astype(np.float32)
        obstacles = env.get_obstacles()

        terminated = False
        truncated = False
        step_idx = 0
        while not (terminated or truncated):
            states_map = {s.agent_id: s for s in env.get_agent_states()}
            obs_low_map = {aid: obs[aid]["low"] for aid in agent_ids}
            actions, relax = baseline.act(states=states_map, obs_low=obs_low_map)
            obs, rewards, terminated, truncated, info = env.step(actions)

            states_next = {s.agent_id: s for s in env.get_agent_states()}
            positions.append(np.stack([states_next[aid].position for aid in agent_ids], axis=0).astype(np.float32))
            unsafe_row = np.asarray([bool(info.get("unsafe_flags", {}).get(aid, False)) for aid in agent_ids], dtype=bool)
            unsafe_trace.append(unsafe_row)
            wind_accel = np.asarray(info.get("wind_accel", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
            mean_relax = float(np.mean([float(np.mean(relax[aid])) if relax[aid].size > 0 else 0.0 for aid in agent_ids]))
            frame_labels.append(
                f"t={len(positions)-1}  wind=({wind_accel[0]:+0.2f}, {wind_accel[1]:+0.2f})  "
                f"|w|={float(np.linalg.norm(wind_accel)):.2f}  relax={mean_relax:.4f}"
            )

            for aid in agent_ids:
                ep_return[aid] += float(rewards[aid])
                reached_any[aid] = bool(reached_any[aid] or bool(info.get("reach_flags", {}).get(aid, False)))
                unsafe_any[aid] = bool(unsafe_any[aid] or bool(info.get("unsafe_flags", {}).get(aid, False)))
                r = np.asarray(relax[aid], dtype=np.float32).reshape(-1)
                if r.size > 0:
                    relax_sum += float(np.sum(r))
                    relax_max = max(relax_max, float(np.max(r)))
                    relax_cnt += int(r.size)
            step_idx += 1
            if int(args.progress_interval) > 0 and (step_idx % int(args.progress_interval) == 0):
                print(
                    f"[progress] episode {ep + 1}/{n_episodes} step={step_idx} "
                    f"terminated={terminated} truncated={truncated} relax={mean_relax:.4f}",
                    flush=True,
                )

        n_agents = max(1, len(agent_ids))
        safe_reach_count = int(sum(1 for aid in agent_ids if reached_any[aid] and not unsafe_any[aid]))
        safe_reach_ratio = float(safe_reach_count / n_agents)
        success_round = int(safe_reach_count == n_agents)
        mean_ret = float(sum(ep_return.values()) / n_agents)

        media_path = ""
        if render_enabled and len(positions) > 0:
            suffix = "gif" if render_gif else "png"
            out_path = run_dir / "media" / f"episode_{ep:03d}.{suffix}"
            media_path = _render_episode(
                renderer=renderer,
                positions=positions,
                unsafe_flags=unsafe_trace,
                frame_labels=frame_labels,
                goals=goals,
                obstacles=obstacles,
                out_path=out_path,
                render_gif=render_gif,
                fps=int(args.fps),
            )

        row = {
            "episode": int(ep),
            "seed": int(seed + ep),
            "steps": int(len(positions)),
            "safe_reach_count": int(safe_reach_count),
            "n_agents": int(n_agents),
            "safe_reach_ratio": float(safe_reach_ratio),
            "success_round": int(success_round),
            "episode_return_mean": float(mean_ret),
            "media_path": media_path,
        }
        rows.append(row)
        total_safe_reach += safe_reach_count
        total_agents += n_agents
        success_rounds += success_round
        mean_return_sum += mean_ret

        print(
            f"[episode {ep + 1}/{n_episodes}] "
            f"safe_reach={safe_reach_count}/{n_agents} "
            f"success={success_round} "
            f"ret={mean_ret:.4f} "
            f"steps={len(positions)}"
        )

    summary = {
        "episodes": int(n_episodes),
        "seed_base": int(seed),
        "overall_safe_reach_ratio": float(total_safe_reach / max(1, total_agents)),
        "success_round_rate": float(success_rounds / max(1, n_episodes)),
        "episode_return_mean": float(mean_return_sum / max(1, n_episodes)),
        "relax_mean": float(relax_sum / max(1, relax_cnt)),
        "relax_max": float(relax_max),
        "run_dir": str(run_dir),
    }

    _write_rows_csv(run_dir / "episode_results.csv", rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("==== GCBF-Style Handcrafted Baseline Summary ====")
    print(f"safe_reach_total: {total_safe_reach}/{total_agents} ({summary['overall_safe_reach_ratio']:.4f})")
    print(f"success_rounds: {success_rounds}/{n_episodes} ({summary['success_round_rate']:.4f})")
    print(f"mean_return: {summary['episode_return_mean']:.4f}")
    print(f"relax_mean: {summary['relax_mean']:.6f}")
    print(f"relax_max: {summary['relax_max']:.6f}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
