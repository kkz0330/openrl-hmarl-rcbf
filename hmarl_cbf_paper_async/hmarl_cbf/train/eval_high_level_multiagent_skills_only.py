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

from hmarl_cbf.control import SyncCoordinator
from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.eval import EpisodeTrace, TrajectoryRenderer
from hmarl_cbf.policies import HighLevelPolicy
from hmarl_cbf.skills import SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.types import AgentState


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-UAV high-level-skills-only simulation without low-level QP or low-level learning."
    )
    parser.add_argument("--config", type=str, default="configs/hmarl_cbf/default_async_onpolicy_gcbfplus.yaml")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint for learned high-level policy.")
    parser.add_argument(
        "--skill-sequences",
        type=str,
        default="",
        help=(
            "Semicolon-separated per-agent skill sequences. "
            "Example: 'accelerate,turn_left,cruise;accelerate,turn_right,cruise'"
        ),
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
    parser.add_argument(
        "--states-json",
        type=str,
        default="",
        help=(
            "Optional JSON array of agent states. "
            "Default is a 2-UAV crossing task: "
            "[{'position':[-2,0],'velocity':[0,0],'goal':[2,0]}, {'position':[0,2],'velocity':[0,0],'goal':[0,-2]}]"
        ),
    )
    parser.add_argument(
        "--obstacles-json",
        type=str,
        default="[]",
        help="Optional JSON array of circular obstacles, e.g. '[{\"center\":[0,0],\"radius\":0.5}]'.",
    )
    parser.add_argument("--sync-mode", type=str, default="", choices=["", "sync", "async"])
    parser.add_argument("--render-gif", action="store_true")
    parser.add_argument("--render-png", action="store_true")
    parser.add_argument("--output", type=str, default="artifacts/hmarl_cbf_paper_async/high_only_multiagent")
    return parser.parse_args()


def _default_states() -> List[Dict[str, Any]]:
    return [
        {"position": [-2.0, 0.0], "velocity": [0.0, 0.0], "goal": [2.0, 0.0]},
        {"position": [0.0, 2.0], "velocity": [0.0, 0.0], "goal": [0.0, -2.0]},
    ]


def _obs_high_vec(obs_high: Any) -> np.ndarray:
    return np.concatenate([obs_high.self_state, obs_high.goal_relative, obs_high.neighbor_summary], axis=0).astype(np.float32)


def _parse_states(raw: str, agent_radius: float) -> List[Dict[str, Any]]:
    payload = _default_states() if not raw.strip() else json.loads(raw)
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("states-json must be a non-empty JSON list")
    states: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"state {idx} must be a JSON object")
        states.append(
            {
                "position": np.asarray(item["position"], dtype=np.float32).reshape(2),
                "velocity": np.asarray(item.get("velocity", [0.0, 0.0]), dtype=np.float32).reshape(2),
                "goal": np.asarray(item["goal"], dtype=np.float32).reshape(2),
                "radius": float(item.get("radius", agent_radius)),
            }
        )
    return states


def _parse_obstacles(raw: str) -> List[Dict[str, Any]]:
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("obstacles-json must be a JSON list")
    obstacles: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"obstacle {idx} must be a JSON object")
        obstacles.append(
            {
                "center": np.asarray(item["center"], dtype=np.float32).reshape(2),
                "radius": float(item["radius"]),
            }
        )
    return obstacles


def _fixed_env(cfg: Dict[str, Any], args: argparse.Namespace) -> tuple[MultiUAV2DEnv, Dict[int, Dict[str, Any]], int]:
    env_cfg = dict(cfg["env"])
    temp_env = MultiUAV2DEnv(**env_cfg)
    states = _parse_states(str(args.states_json), agent_radius=float(temp_env.agent_radius))
    obstacles = _parse_obstacles(str(args.obstacles_json))
    env_cfg["n_agents"] = len(states)
    env_cfg["n_obstacles"] = len(obstacles)
    env = MultiUAV2DEnv(**env_cfg)
    obs, _ = env.reset(seed=int(args.seed), options={"states": states, "obstacles": obstacles})
    env._preset_obs = obs  # type: ignore[attr-defined]
    return env, obs, len(states)


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


def _parse_manual_sequences(raw: str, skill_names: List[str], n_agents: int) -> Dict[int, List[int]]:
    if not raw.strip():
        return {}
    chunks = [chunk.strip() for chunk in raw.split(";")]
    if len(chunks) != n_agents:
        raise ValueError(f"skill-sequences must provide exactly {n_agents} agent sequences separated by ';'")
    name_to_id = {name: idx for idx, name in enumerate(skill_names)}
    out: Dict[int, List[int]] = {}
    for agent_id, chunk in enumerate(chunks):
        tokens = [item.strip() for item in chunk.split(",") if item.strip()]
        seq: List[int] = []
        for token in tokens:
            if token.isdigit():
                seq.append(int(token))
                continue
            if token not in name_to_id:
                raise ValueError(f"unknown skill '{token}', available: {skill_names}")
            seq.append(int(name_to_id[token]))
        out[agent_id] = seq
    return out


def _sample_skill(policy: HighLevelPolicy, obs_high: Any, deterministic: bool) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is required to sample high-level skills from checkpoint")
    obs_vec = _obs_high_vec(obs_high)
    with torch.no_grad():
        out = policy.act(torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0), deterministic=deterministic)
    return int(out["z"].detach().cpu().numpy()[0])


def _list_valid_skill_ids(runtime: SkillRuntimeManager, agent_id: int, state: AgentState) -> List[int]:
    valid: List[int] = []
    for skill_id, skill in runtime.skill_by_id.items():
        ctx = runtime._prepare_agent_ctx(agent_id=agent_id, skill=skill, state=state)  # type: ignore[attr-defined]
        if skill.initiation_set_fn(state, ctx):
            valid.append(int(skill_id))
    return sorted(valid)


def _sample_valid_skill(
    policy: HighLevelPolicy,
    obs_high: Any,
    valid_skill_ids: List[int],
    deterministic: bool,
) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is required to sample high-level skills from checkpoint")
    if not valid_skill_ids:
        raise ValueError("valid_skill_ids must be non-empty")
    obs_vec = _obs_high_vec(obs_high)
    with torch.no_grad():
        logits, _ = policy.forward(torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0))
        masked_logits = torch.full_like(logits, -1e9)
        masked_logits[:, valid_skill_ids] = logits[:, valid_skill_ids]
        if deterministic:
            z = torch.argmax(masked_logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=masked_logits)
            z = dist.sample()
    return int(z.detach().cpu().item())


def main() -> None:
    args = _parse_args()
    cfg: Dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    env, obs, n_agents = _fixed_env(cfg, args)
    agent_ids = sorted(obs.keys())
    obs_high = obs[agent_ids[0]]["high"]
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
    skill_runtime.reset(agent_ids)

    manual_sequences = _parse_manual_sequences(str(args.skill_sequences), skill_names, n_agents)
    seq_cursors = {aid: 0 for aid in agent_ids}
    high_policy = None
    if str(args.checkpoint).strip():
        high_policy = _build_high_policy_from_checkpoint(
            checkpoint_path=Path(args.checkpoint),
            obs_dim=obs_dim_high,
            n_skills=len(skills),
            hidden_dim=int(cfg["model"]["high_hidden_dim"]),
        )

    if not manual_sequences and high_policy is None:
        raise ValueError("Provide either --skill-sequences or --checkpoint")

    mode = str(args.sync_mode).strip().lower() or str(cfg["synchronization"].get("mode", "async")).strip().lower()
    coordinator = SyncCoordinator(
        num_agents=n_agents,
        t_sync_max=int(cfg["synchronization"]["t_sync_max"]),
        mode=mode,
    )
    coordinator.reset()

    def pick_next_skill(agent_id: int, step_obs: Dict[int, Dict[str, Any]], current_skill_id: int | None) -> int | None:
        manual_seq = manual_sequences.get(agent_id, [])
        if seq_cursors[agent_id] < len(manual_seq):
            skill_id = int(manual_seq[seq_cursors[agent_id]])
            seq_cursors[agent_id] += 1
            return skill_id
        if high_policy is None:
            policy = str(args.sequence_end_policy).strip().lower()
            if policy == "repeat_last":
                if current_skill_id is None:
                    raise RuntimeError(f"agent {agent_id}: cannot repeat last skill before any skill was activated")
                return int(current_skill_id)
            if policy == "stop":
                return None
            raise RuntimeError(f"agent {agent_id}: manual skill sequence exhausted and no high-level checkpoint provided")
        step_states = {s.agent_id: s for s in env.get_agent_states()}
        valid_skill_ids = _list_valid_skill_ids(skill_runtime, agent_id=agent_id, state=step_states[agent_id])
        return _sample_valid_skill(
            high_policy,
            step_obs[agent_id]["high"],
            valid_skill_ids,
            deterministic=bool(args.deterministic),
        )

    states0 = {s.agent_id: s for s in env.get_agent_states()}
    current_skill_ids: Dict[int, int] = {}
    chosen_skills_by_agent: Dict[int, List[Dict[str, Any]]] = {aid: [] for aid in agent_ids}
    for aid in agent_ids:
        skill_id = pick_next_skill(aid, obs, None)
        if skill_id is None:
            raise ValueError(f"agent {aid}: no initial skill available")
        try:
            skill_runtime.activate_skill(agent_id=aid, skill_id=int(skill_id), state=states0[aid])
        except ValueError as exc:
            valid = _list_valid_skill_ids(skill_runtime, agent_id=aid, state=states0[aid])
            valid_names = [skill_names[idx] for idx in valid]
            raise ValueError(
                f"agent {aid}: cannot activate initial skill '{skill_names[int(skill_id)]}', valid choices={valid_names}"
            ) from exc
        current_skill_ids[aid] = int(skill_id)
        chosen_skills_by_agent[aid].append({"t": 0, "skill_id": int(skill_id), "skill_name": skill_names[int(skill_id)]})

    trace_positions: List[np.ndarray] = []
    trace_unsafe: List[np.ndarray] = []
    frame_labels: List[str] = []
    exhausted_agents: List[int] = []
    stop_reason = "running"
    terminated = False
    truncated = False
    info: Dict[str, Any] = {"reach_flags": {}, "unsafe_flags": {}}
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(cfg["env"]["horizon"])

    for t in range(max_steps):
        states = {s.agent_id: s for s in env.get_agent_states()}
        actions: Dict[int, np.ndarray] = {}
        for aid in agent_ids:
            target = skill_runtime.control_target(agent_id=aid, state=states[aid], obs_low=obs[aid]["low"])
            actions[aid] = np.asarray(target["u_ref_skill"], dtype=np.float32).reshape(2)

        next_obs, rewards, terminated, truncated, info = env.step(actions)
        next_states = {s.agent_id: s for s in env.get_agent_states()}
        skill_out = skill_runtime.step_all(
            states=next_states,
            obs_low={aid: next_obs[aid]["low"] for aid in agent_ids},
            executed_actions=actions,
        )
        beta = {aid: bool(skill_out[aid].beta) for aid in agent_ids}
        sync_res = coordinator.step(beta)

        trace_positions.append(
            np.stack([np.asarray(next_states[aid].position, dtype=np.float32).reshape(2) for aid in agent_ids], axis=0)
        )
        trace_unsafe.append(
            np.asarray([bool(info.get("unsafe_flags", {}).get(aid, False)) for aid in agent_ids], dtype=bool)
        )
        labels = [f"a{aid}:{skill_names[current_skill_ids[aid]]}:{float(rewards[aid]):+.2f}" for aid in agent_ids]
        frame_labels.append(f"t={t} | " + " | ".join(labels))
        obs = next_obs

        if sync_res.sync_switch and not (terminated or truncated):
            for aid in sync_res.switch_agents:
                next_skill = pick_next_skill(aid, obs, current_skill_ids.get(aid))
                if next_skill is None:
                    exhausted_agents.append(aid)
                    stop_reason = "manual_sequence_exhausted"
                    continue
                try:
                    skill_runtime.activate_skill(agent_id=aid, skill_id=int(next_skill), state=next_states[aid])
                except ValueError as exc:
                    valid = _list_valid_skill_ids(skill_runtime, agent_id=aid, state=next_states[aid])
                    valid_names = [skill_names[idx] for idx in valid]
                    raise ValueError(
                        f"agent {aid}: cannot activate skill '{skill_names[int(next_skill)]}' at t={t + 1}, "
                        f"valid choices={valid_names}"
                    ) from exc
                current_skill_ids[aid] = int(next_skill)
                chosen_skills_by_agent[aid].append(
                    {"t": t + 1, "skill_id": int(next_skill), "skill_name": skill_names[int(next_skill)]}
                )
            if exhausted_agents:
                break

        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    goals = np.stack([np.asarray(s.goal, dtype=np.float32).reshape(2) for s in env.get_agent_states()], axis=0)
    obstacles = env.get_obstacles()

    media_path = ""
    if trace_positions and (bool(args.render_gif) or bool(args.render_png)):
        renderer = TrajectoryRenderer(world_size=float(getattr(env, "world_size", 10.0)))
        trace = EpisodeTrace(
            positions=np.stack(trace_positions, axis=0),
            goals=goals,
            obstacles=obstacles,
            unsafe_flags=np.stack(trace_unsafe, axis=0),
            frame_labels=frame_labels,
        )
        if bool(args.render_gif) or not bool(args.render_png):
            media_path = renderer.render_gif(trace, out_dir / "multi_uav_high_only.gif", fps=8)
        else:
            media_path = renderer.render_static(trace, out_dir / "multi_uav_high_only.png")

    final_states = {s.agent_id: s for s in env.get_agent_states()}
    agent_summaries: Dict[str, Any] = {}
    team_success = True
    for aid in agent_ids:
        reached = bool(info.get("reach_flags", {}).get(aid, False)) if trace_positions else False
        unsafe = bool(info.get("unsafe_flags", {}).get(aid, False)) if trace_positions else False
        if not reached or unsafe:
            team_success = False
        final_state = final_states[aid]
        agent_summaries[str(aid)] = {
            "reached_goal": bool(reached),
            "unsafe": bool(unsafe),
            "final_position": np.asarray(final_state.position, dtype=np.float32).reshape(2).tolist(),
            "final_velocity": np.asarray(final_state.velocity, dtype=np.float32).reshape(2).tolist(),
            "goal": np.asarray(final_state.goal, dtype=np.float32).reshape(2).tolist(),
            "chosen_skills": chosen_skills_by_agent[aid],
            "manual_sequence_provided": [skill_names[idx] for idx in manual_sequences.get(aid, [])],
        }

    summary = {
        "n_agents": int(n_agents),
        "sync_mode": str(mode),
        "steps": int(len(trace_positions)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "exhausted_agents": exhausted_agents,
        "stop_reason": str(stop_reason),
        "team_success": bool(team_success),
        "agent_summaries": agent_summaries,
        "obstacles": [
            {
                "center": np.asarray(obs_item["center"], dtype=np.float32).reshape(2).tolist(),
                "radius": float(obs_item["radius"]),
            }
            for obs_item in obstacles
        ],
        "sequence_end_policy": str(args.sequence_end_policy),
        "media_path": str(media_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
