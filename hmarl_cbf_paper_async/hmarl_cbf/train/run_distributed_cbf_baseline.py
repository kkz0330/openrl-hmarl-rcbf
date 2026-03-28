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

from hmarl_cbf.baselines import DistributedCBFBaselineConfig, DistributedCBFBaselineController
from hmarl_cbf.control import ConstraintBuilder, DifferentiableQPSolver
from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.eval import EpisodeTrace, TrajectoryRenderer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-parameter distributed CBF-QP baseline (GCBF+ style) with visualization."
    )
    parser.add_argument("--config", type=str, default="configs/hmarl_cbf/baseline_distributed_cbf.yaml")
    parser.add_argument("--output-root", type=str, default="artifacts/hmarl_cbf_baseline")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--render-gif", action="store_true", help="Render each episode to GIF.")
    parser.add_argument("--render-png", action="store_true", help="Render each episode to static PNG.")
    parser.add_argument("--fps", type=int, default=8, help="GIF fps.")
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
        out = output_root / time.strftime("baseline_%Y%m%d_%H%M%S")
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
    )
    if render_gif:
        return renderer.render_gif(trace, out_path, fps=max(1, int(fps)))
    return renderer.render_static(trace, out_path)


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    seed = int(cfg.get("seed", 42) if int(args.seed) < 0 else int(args.seed))
    _seed_all(seed)

    env_cfg = dict(cfg["env"])
    env = MultiUAV2DEnv(**env_cfg)

    action_limit = float(env_cfg["action_limit"])
    constraint_builder = ConstraintBuilder(
        d_min_agent=float(cfg["safety"]["d_min_agent"]),
        d_safe_obs=float(cfg["safety"]["d_safe_obs"]),
        u_min=[-action_limit, -action_limit],
        u_max=[action_limit, action_limit],
    )
    qp_solver = DifferentiableQPSolver(
        action_dim=2,
        use_stub_if_unavailable=bool(cfg["qp"].get("use_stub_if_unavailable", True)),
        ecos_max_iters=int(cfg["qp"].get("ecos_max_iters", 500)),
        scs_max_iters=int(cfg["qp"].get("scs_max_iters", 10_000)),
        scs_eps=float(cfg["qp"].get("scs_eps", 1e-4)),
    )
    baseline_cfg = DistributedCBFBaselineConfig.from_mapping(cfg["baseline"])
    baseline = DistributedCBFBaselineController(
        constraint_builder=constraint_builder,
        qp_solver=qp_solver,
        config=baseline_cfg,
    )

    run_dir = _make_run_dir(Path(args.output_root), args.run_name)
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    renderer = TrajectoryRenderer(world_size=float(env_cfg["world_size"]))

    rows: List[Dict[str, Any]] = []
    total_safe_reach = 0
    total_agents = 0
    success_rounds = 0
    mean_return_sum = 0.0
    feasible_sum = 0.0
    feasible_cnt = 0

    render_gif = bool(args.render_gif) or not bool(args.render_png)
    n_episodes = max(1, int(args.episodes))
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        _ = obs  # observation is not used by this fixed baseline
        agent_ids = sorted(obs.keys())
        reached_any = {aid: False for aid in agent_ids}
        unsafe_any = {aid: False for aid in agent_ids}
        ep_return = {aid: 0.0 for aid in agent_ids}
        positions: List[np.ndarray] = []
        unsafe_trace: List[np.ndarray] = []
        goals = np.stack([s.goal for s in env.get_agent_states()], axis=0).astype(np.float32)
        obstacles = env.get_obstacles()

        terminated = False
        truncated = False
        while not (terminated or truncated):
            states_map = {s.agent_id: s for s in env.get_agent_states()}
            actions, solutions = baseline.solve_batch(states=states_map, obstacles=obstacles)
            _, rewards, terminated, truncated, info = env.step(actions)

            states_next = {s.agent_id: s for s in env.get_agent_states()}
            positions.append(np.stack([states_next[aid].position for aid in agent_ids], axis=0).astype(np.float32))
            unsafe_row = np.asarray([bool(info.get("unsafe_flags", {}).get(aid, False)) for aid in agent_ids], dtype=bool)
            unsafe_trace.append(unsafe_row)

            for aid in agent_ids:
                ep_return[aid] += float(rewards[aid])
                reached_any[aid] = bool(reached_any[aid] or bool(info.get("reach_flags", {}).get(aid, False)))
                unsafe_any[aid] = bool(unsafe_any[aid] or bool(info.get("unsafe_flags", {}).get(aid, False)))
                feasible_sum += 1.0 if bool(solutions[aid].feasible) else 0.0
                feasible_cnt += 1

        n_agents = max(1, len(agent_ids))
        safe_reach_count = int(sum(1 for aid in agent_ids if reached_any[aid] and not unsafe_any[aid]))
        safe_reach_ratio = float(safe_reach_count / n_agents)
        success_round = int(safe_reach_count == n_agents)
        mean_ret = float(sum(ep_return.values()) / n_agents)

        media_path = ""
        if len(positions) > 0:
            suffix = "gif" if render_gif else "png"
            out_path = run_dir / "media" / f"episode_{ep:03d}.{suffix}"
            media_path = _render_episode(
                renderer=renderer,
                positions=positions,
                unsafe_flags=unsafe_trace,
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
        "qp_feasible_rate": float(feasible_sum / max(1, feasible_cnt)),
        "run_dir": str(run_dir),
    }

    _write_rows_csv(run_dir / "episode_results.csv", rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("==== Baseline Summary ====")
    print(f"safe_reach_total: {total_safe_reach}/{total_agents} ({summary['overall_safe_reach_ratio']:.4f})")
    print(f"success_rounds: {success_rounds}/{n_episodes} ({summary['success_round_rate']:.4f})")
    print(f"mean_return: {summary['episode_return_mean']:.4f}")
    print(f"qp_feasible_rate: {summary['qp_feasible_rate']:.4f}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()

