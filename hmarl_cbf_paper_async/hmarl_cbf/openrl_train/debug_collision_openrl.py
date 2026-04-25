from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from hmarl_cbf.env.obstacles import normalize_obstacle, obstacle_bounding_radius, obstacle_surface_distance
from hmarl_cbf.openrl_agents import HighMAPPOAgent, HighMAPPOAgentConfig
from hmarl_cbf.openrl_train.common import (
    apply_config_section_defaults,
    JointLowLevelPolicyExecutor,
    build_core_env,
    build_high_env,
    build_high_net,
    build_low_agent,
    build_low_env,
    load_yaml,
    save_json,
)
from hmarl_cbf.skills import build_default_skill_library


def _build_agents_and_env(cfg: Dict[str, Any], args: argparse.Namespace):
    core = build_core_env(cfg)
    low_env = build_low_env(cfg, core, torch_device=args.torch_device)
    low_agent = build_low_agent(low_env, cfg, torch_device=args.torch_device)
    low_agent.load(args.low_checkpoint)

    executor = JointLowLevelPolicyExecutor(
        low_agent,
        core,
        n_skills=len(build_default_skill_library()),
        include_local_low_obs_in_critic=bool(low_env.config.include_local_obs_in_critic),
        deterministic=bool(args.deterministic),
    )
    env = build_high_env(cfg, core, low_level_executor=executor)
    net = build_high_net(env, torch_device=args.torch_device)
    net.reset()
    high_agent = HighMAPPOAgent(net, HighMAPPOAgentConfig())
    high_agent.load(args.high_checkpoint)
    return env, high_agent


def _sample_segment_overlap(
    start: np.ndarray,
    end: np.ndarray,
    *,
    agent_radius: float,
    obstacle: Dict[str, Any],
    probe_step: float,
) -> Dict[str, Any] | None:
    seg = np.asarray(end, dtype=np.float32).reshape(2) - np.asarray(start, dtype=np.float32).reshape(2)
    seg_len = float(np.linalg.norm(seg))
    if seg_len <= 1e-8:
        return None

    obs_norm = normalize_obstacle(obstacle)
    bound_r = max(float(obstacle_bounding_radius(obs_norm)), 0.0)
    local_probe_step = max(1e-3, min(float(probe_step), max(seg_len / 2.0, 1e-3)))
    n_samples = max(2, int(np.ceil(seg_len / local_probe_step)) + 1)

    best_idx = -1
    best_clearance = float("inf")
    best_point = None
    for idx in range(1, n_samples):
        alpha = float(idx / n_samples)
        point = (1.0 - alpha) * np.asarray(start, dtype=np.float32) + alpha * np.asarray(end, dtype=np.float32)
        clearance = float(obstacle_surface_distance(point, obs_norm) - agent_radius)
        if clearance < best_clearance:
            best_clearance = clearance
            best_idx = idx
            best_point = point
        if clearance <= 0.0:
            return {
                "alpha": alpha,
                "sample_index": int(idx),
                "sample_count": int(n_samples),
                "sample_point": np.asarray(point, dtype=np.float32).reshape(2).tolist(),
                "clearance": clearance,
                "obstacle_bound_radius": bound_r,
            }

    if best_point is None:
        return None
    return {
        "alpha": float(best_idx / n_samples),
        "sample_index": int(best_idx),
        "sample_count": int(n_samples),
        "sample_point": np.asarray(best_point, dtype=np.float32).reshape(2).tolist(),
        "clearance": float(best_clearance),
        "obstacle_bound_radius": bound_r,
    }


def _debug_episode(
    env,
    high_agent,
    *,
    episode_index: int,
    seed: int,
    deterministic: bool,
    probe_step: float,
    max_print: int,
) -> Dict[str, Any]:
    high_agent.reset()
    obs, infos = env.reset(seed=seed)

    agent_ids = list(env.possible_agents)
    agent_radius = float(env.core_env.env.agent_radius)
    prev_positions = {
        int(aid): np.asarray(env.core_env.get_states()[aid].position, dtype=np.float32).reshape(2).copy()
        for aid in agent_ids
    }

    anomaly_rows: List[Dict[str, Any]] = []
    printed = 0
    physical_steps = 0

    while env.agents:
        obs, rewards, terminations, truncations, infos, _ = high_agent.step_env(
            env,
            obs,
            infos,
            deterministic=deterministic,
        )

        for step_record in env.last_low_step_records:
            physical_steps += 1
            unsafe_flags = dict(step_record["unsafe_flags"])
            positions = dict(step_record["positions"])
            step_count = int(step_record.get("step_count", physical_steps))
            obstacles = [dict(obs_item) for obs_item in env.core_env.get_obstacles()]

            for aid in agent_ids:
                curr = np.asarray(positions[int(aid)], dtype=np.float32).reshape(2)
                prev = np.asarray(prev_positions[int(aid)], dtype=np.float32).reshape(2)
                unsafe = bool(unsafe_flags.get(int(aid), False))

                for obs_idx, obstacle in enumerate(obstacles):
                    obs_norm = normalize_obstacle(obstacle)
                    clearance_now = float(obstacle_surface_distance(curr, obs_norm) - agent_radius)
                    inside_now = bool(clearance_now <= 0.0)
                    segment_probe = _sample_segment_overlap(
                        prev,
                        curr,
                        agent_radius=agent_radius,
                        obstacle=obs_norm,
                        probe_step=probe_step,
                    )
                    segment_overlap = bool(segment_probe is not None and float(segment_probe["clearance"]) <= 0.0)

                    if not unsafe and (inside_now or segment_overlap):
                        row = {
                            "episode": int(episode_index),
                            "seed": int(seed),
                            "step_count": int(step_count),
                            "agent_id": int(aid),
                            "unsafe_flag": bool(unsafe),
                            "obstacle_index": int(obs_idx),
                            "obstacle": {
                                k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in obs_norm.items()
                            },
                            "prev_position": prev.tolist(),
                            "curr_position": curr.tolist(),
                            "clearance_now": float(clearance_now),
                            "inside_now": bool(inside_now),
                            "segment_overlap": bool(segment_overlap),
                            "segment_probe": segment_probe,
                        }
                        anomaly_rows.append(row)
                        if printed < max_print:
                            print("collision_debug_anomaly", json.dumps(row, ensure_ascii=False))
                            printed += 1

                prev_positions[int(aid)] = curr.copy()

        if not env.agents or all(bool(terminations[aid] or truncations[aid]) for aid in terminations):
            break

    return {
        "episode": int(episode_index),
        "seed": int(seed),
        "physical_steps": int(physical_steps),
        "anomaly_count": int(len(anomaly_rows)),
        "anomalies": anomaly_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug collision mismatches for OpenRL stack checkpoints.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--low-checkpoint", type=str, required=True)
    parser.add_argument("--high-checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-device", type=str, default="cpu")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--probe-step",
        type=float,
        default=0.01,
        help="Sampling step for segment overlap probing; smaller is stricter but slower.",
    )
    parser.add_argument("--max-print", type=int, default=20)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    args = apply_config_section_defaults(
        args,
        cfg,
        section="openrl_eval",
        defaults={
            "low_checkpoint": "",
            "high_checkpoint": "",
            "episodes": 3,
            "seed": 42,
            "torch_device": "cpu",
            "deterministic": True,
            "output_json": "",
        },
    )

    env, high_agent = _build_agents_and_env(cfg, args)
    all_rows: List[Dict[str, Any]] = []
    total_anomalies = 0

    for ep in range(int(args.episodes)):
        row = _debug_episode(
            env,
            high_agent,
            episode_index=ep,
            seed=int(args.seed) + ep,
            deterministic=bool(args.deterministic),
            probe_step=float(args.probe_step),
            max_print=int(args.max_print),
        )
        all_rows.append(row)
        total_anomalies += int(row["anomaly_count"])
        print(
            f"collision_debug_episode={ep + 1} "
            f"seed={int(row['seed'])} "
            f"steps={int(row['physical_steps'])} "
            f"anomalies={int(row['anomaly_count'])}"
        )

    summary = {
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "total_anomalies": int(total_anomalies),
        "episodes_with_anomalies": int(sum(1 for row in all_rows if int(row["anomaly_count"]) > 0)),
        "rows": all_rows,
    }

    print("collision_debug_total_anomalies", int(total_anomalies))
    if args.output_json:
        save_json(args.output_json, summary)


if __name__ == "__main__":
    main()
