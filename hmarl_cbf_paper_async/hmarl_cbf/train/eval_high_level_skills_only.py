from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime entrypoint
    raise RuntimeError("PyYAML is required to load config") from exc

from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.eval import EpisodeTrace, TrajectoryRenderer
from hmarl_cbf.policies import HighLevelPolicy
from hmarl_cbf.skills import SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.types import AgentState


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-UAV high-level-skill-only simulation without low-level QP or low-level learning."
    )
    parser.add_argument("--config", type=str, default="configs/hmarl_cbf/default_async_onpolicy_gcbfplus.yaml")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint for learned high-level policy.")
    parser.add_argument(
        "--skill-sequence",
        type=str,
        default="",
        help="Comma-separated manual skill names or ids, e.g. accelerate,turn_left,cruise,decelerate",
    )
    parser.add_argument(
        "--sequence-end-policy",
        type=str,
        default="stop",
        choices=["stop", "repeat_last", "error"],
        help="What to do when a manual skill sequence is exhausted and no checkpoint is provided.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--start-x", type=float, default=-4.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--goal-x", type=float, default=4.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--obstacle-x", type=float, default=0.0)
    parser.add_argument("--obstacle-y", type=float, default=0.0)
    parser.add_argument("--obstacle-r", type=float, default=1.0)
    parser.add_argument("--init-vx", type=float, default=0.0)
    parser.add_argument("--init-vy", type=float, default=0.0)
    parser.add_argument("--render-gif", action="store_true")
    parser.add_argument("--render-png", action="store_true")
    parser.add_argument("--output", type=str, default="artifacts/hmarl_cbf_paper_async/high_only")
    return parser.parse_args()


def _obs_high_vec(obs_high: Any) -> np.ndarray:
    return np.concatenate([obs_high.self_state, obs_high.goal_relative, obs_high.neighbor_summary], axis=0).astype(np.float32)


def _fixed_env(cfg: Dict[str, Any], args: argparse.Namespace) -> MultiUAV2DEnv:
    env_cfg = dict(cfg["env"])
    env_cfg["n_agents"] = 1
    env_cfg["n_obstacles"] = 1
    env = MultiUAV2DEnv(**env_cfg)
    states = [
        {
            "position": np.asarray([args.start_x, args.start_y], dtype=np.float32),
            "velocity": np.asarray([args.init_vx, args.init_vy], dtype=np.float32),
            "goal": np.asarray([args.goal_x, args.goal_y], dtype=np.float32),
            "radius": float(env.agent_radius),
        }
    ]
    obstacles = [{"center": np.asarray([args.obstacle_x, args.obstacle_y], dtype=np.float32), "radius": float(args.obstacle_r)}]
    obs, _ = env.reset(seed=int(args.seed), options={"states": states, "obstacles": obstacles})
    env._preset_obs = obs  # type: ignore[attr-defined]
    return env


def _build_high_policy_from_checkpoint(
    checkpoint_path: Path,
    obs_dim: int,
    n_skills: int,
    hidden_dim: int,
) -> HighLevelPolicy:
    if torch is None:
        raise RuntimeError("PyTorch is required to load a high-level checkpoint")
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload["high_policy"]
    policy = HighLevelPolicy(obs_dim=obs_dim, n_skills=n_skills, hidden_dim=hidden_dim)
    policy.load_state_dict(state, strict=False)
    policy.eval()
    return policy


def _parse_manual_sequence(raw: str, skill_names: List[str]) -> List[int]:
    if not raw.strip():
        return []
    out: List[int] = []
    name_to_id = {name: idx for idx, name in enumerate(skill_names)}
    for token in [item.strip() for item in raw.split(",") if item.strip()]:
        if token.isdigit():
            out.append(int(token))
            continue
        if token not in name_to_id:
            raise ValueError(f"unknown skill '{token}', available: {skill_names}")
        out.append(int(name_to_id[token]))
    return out


def _sample_skill(policy: HighLevelPolicy, obs_high: Any, deterministic: bool) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is required to sample high-level skills from checkpoint")
    obs_vec = _obs_high_vec(obs_high)
    with torch.no_grad():
        out = policy.act(torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0), deterministic=deterministic)
    return int(out["z"].detach().cpu().numpy()[0])


def main() -> None:
    args = _parse_args()
    cfg: Dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    env = _fixed_env(cfg, args)
    obs = getattr(env, "_preset_obs")
    agent_id = 0
    obs_high = obs[agent_id]["high"]
    obs_low = obs[agent_id]["low"]
    obs_dim_high = int(obs_high.self_state.shape[0] + obs_high.goal_relative.shape[0] + obs_high.neighbor_summary.shape[0])

    skills = build_default_skill_library(max_duration=int(cfg["skills"]["default_max_duration"]))
    skill_names = [s.name for s in skills]
    skill_runtime = SkillRuntimeManager(
        skills,
        default_ctx={
            **dict(cfg["skills"]["params"]),
            "action_limit": float(cfg["env"]["action_limit"]),
            "cbf_u_max": float(cfg["env"]["action_limit"]),
            "dt": float(cfg["env"]["dt"]),
        },
    )
    skill_runtime.reset([agent_id])

    manual_sequence = _parse_manual_sequence(str(args.skill_sequence), skill_names)
    high_policy = None
    if str(args.checkpoint).strip():
        high_policy = _build_high_policy_from_checkpoint(
            checkpoint_path=Path(args.checkpoint),
            obs_dim=obs_dim_high,
            n_skills=len(skills),
            hidden_dim=int(cfg["model"]["high_hidden_dim"]),
        )

    if not manual_sequence and high_policy is None:
        raise ValueError("Provide either --skill-sequence or --checkpoint")

    def pick_next_skill(step_obs: Dict[int, Dict[str, Any]], seq_cursor: int, current_skill_id: int | None) -> tuple[int | None, int]:
        if seq_cursor < len(manual_sequence):
            return int(manual_sequence[seq_cursor]), seq_cursor + 1
        if high_policy is None:
            policy = str(args.sequence_end_policy).strip().lower()
            if policy == "repeat_last":
                if current_skill_id is None:
                    raise RuntimeError("cannot repeat last skill because no skill has been activated yet")
                return int(current_skill_id), seq_cursor
            if policy == "stop":
                return None, seq_cursor
            raise RuntimeError("manual skill sequence exhausted and no high-level checkpoint provided")
        sid = _sample_skill(high_policy, step_obs[agent_id]["high"], deterministic=bool(args.deterministic))
        return sid, seq_cursor

    seq_cursor = 0
    states0 = {s.agent_id: s for s in env.get_agent_states()}
    current_skill, seq_cursor = pick_next_skill(obs, seq_cursor, None)
    if current_skill is None:
        raise ValueError("no initial skill available; provide --skill-sequence or --checkpoint")
    skill_runtime.activate_skill(agent_id=agent_id, skill_id=int(current_skill), state=states0[agent_id])

    trace_positions: List[np.ndarray] = []
    trace_unsafe: List[np.ndarray] = []
    frame_labels: List[str] = []
    chosen_skills: List[Dict[str, Any]] = [{"t": 0, "skill_id": int(current_skill), "skill_name": skill_names[int(current_skill)]}]
    sequence_exhausted = False
    stop_reason = "running"
    terminated = False
    truncated = False
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(cfg["env"]["horizon"])

    for t in range(max_steps):
        states = {s.agent_id: s for s in env.get_agent_states()}
        obs_low = obs[agent_id]["low"]
        target = skill_runtime.control_target(agent_id=agent_id, state=states[agent_id], obs_low=obs_low)
        action = np.asarray(target["u_ref_skill"], dtype=np.float32).reshape(2)

        next_obs, rewards, terminated, truncated, info = env.step({agent_id: action})
        next_states = {s.agent_id: s for s in env.get_agent_states()}
        step_out = skill_runtime.step(
            agent_id=agent_id,
            state=next_states[agent_id],
            obs_low=next_obs[agent_id]["low"],
            executed_action=action,
        )

        trace_positions.append(np.asarray(next_states[agent_id].position, dtype=np.float32).reshape(1, 2))
        trace_unsafe.append(np.asarray([bool(info.get("unsafe_flags", {}).get(agent_id, False))], dtype=bool))
        frame_labels.append(f"t={t} skill={skill_names[int(current_skill)]} reward={float(rewards[agent_id]):.3f}")
        obs = next_obs

        if bool(step_out.beta) and not (terminated or truncated):
            next_skill, seq_cursor = pick_next_skill(obs, seq_cursor, int(current_skill))
            if next_skill is None:
                sequence_exhausted = True
                stop_reason = "manual_sequence_exhausted"
                break
            current_skill = int(next_skill)
            states_round = {s.agent_id: s for s in env.get_agent_states()}
            skill_runtime.activate_skill(agent_id=agent_id, skill_id=int(current_skill), state=states_round[agent_id])
            chosen_skills.append({"t": t + 1, "skill_id": int(current_skill), "skill_name": skill_names[int(current_skill)]})

        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    goals = np.stack([np.asarray(s.goal, dtype=np.float32).reshape(2) for s in env.get_agent_states()], axis=0)
    obstacles = env.get_obstacles()

    media_path = ""
    if len(trace_positions) > 0 and (bool(args.render_gif) or bool(args.render_png)):
        renderer = TrajectoryRenderer(world_size=float(getattr(env, "world_size", 10.0)))
        trace = EpisodeTrace(
            positions=np.stack(trace_positions, axis=0),
            goals=goals,
            obstacles=obstacles,
            unsafe_flags=np.stack(trace_unsafe, axis=0),
            frame_labels=frame_labels,
        )
        if bool(args.render_gif) or not bool(args.render_png):
            media_path = renderer.render_gif(trace, out_dir / "single_uav_high_only.gif", fps=8)
        else:
            media_path = renderer.render_static(trace, out_dir / "single_uav_high_only.png")

    final_state: AgentState = env.get_agent_states()[0]
    reached = bool(info.get("reach_flags", {}).get(agent_id, False)) if len(trace_positions) > 0 else False
    unsafe = bool(info.get("unsafe_flags", {}).get(agent_id, False)) if len(trace_positions) > 0 else False
    summary = {
        "steps": int(len(trace_positions)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "sequence_exhausted": bool(sequence_exhausted),
        "stop_reason": str(stop_reason),
        "reached_goal": bool(reached),
        "unsafe": bool(unsafe),
        "final_position": np.asarray(final_state.position, dtype=np.float32).reshape(2).tolist(),
        "final_velocity": np.asarray(final_state.velocity, dtype=np.float32).reshape(2).tolist(),
        "goal": np.asarray(final_state.goal, dtype=np.float32).reshape(2).tolist(),
        "chosen_skills": chosen_skills,
        "manual_sequence_provided": [skill_names[idx] for idx in manual_sequence],
        "sequence_end_policy": str(args.sequence_end_policy),
        "media_path": str(media_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
