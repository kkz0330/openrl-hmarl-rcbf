from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime entrypoint
    raise RuntimeError("PyTorch is required to run checkpoint evaluation") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime entrypoint
    raise RuntimeError("PyYAML is required to load config") from exc

from hmarl_cbf.buffer import HierRolloutBuffer
from hmarl_cbf.control import ConstraintBuilder, DifferentiableQPSolver, LowLevelSafeController, SyncCoordinator
from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.policies import HighLevelPolicy, LowLevelQPPolicy
from hmarl_cbf.skills import SKILL_HOVER, SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.train import TrainerSyncOnPolicy
from hmarl_cbf.types import AgentObsHigh, AgentState


def _load_state_dict_flexible(module: torch.nn.Module, state: Dict[str, Any], name: str) -> None:
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[warn] {name} state_dict partial load: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HMARL-CBF checkpoint on random scenes.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoints/last.pt")
    parser.add_argument("--config", type=str, default="", help="Optional config yaml path. Overrides checkpoint config.")
    parser.add_argument("--output-root", type=str, default="artifacts/hmarl_cbf_eval")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=30.0, help="Per-episode simulated duration in seconds.")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed; scene seed = base + episode index.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic high-level skill selection.")
    return parser.parse_args()


def _resolve_latest_checkpoint() -> Path:
    latest_marker = Path("artifacts/hmarl_cbf/LATEST_RUN")
    if not latest_marker.exists():
        raise FileNotFoundError("No checkpoint path provided and artifacts/hmarl_cbf/LATEST_RUN not found")
    run_dir = Path(latest_marker.read_text(encoding="utf-8").strip())
    ckpt = run_dir / "checkpoints" / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt}")
    return ckpt


def _load_config(args: argparse.Namespace, ckpt_payload: Dict[str, Any], ckpt_path: Path) -> Dict[str, Any]:
    if args.config:
        return yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if "config" in ckpt_payload and isinstance(ckpt_payload["config"], dict):
        return dict(ckpt_payload["config"])
    snapshot = ckpt_path.parents[1] / "config_snapshot.yaml"
    if snapshot.exists():
        return yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    raise RuntimeError("Cannot resolve config from --config, checkpoint payload, or config_snapshot.yaml")


def _make_output_dir(output_root: Path, run_name: str) -> Path:
    if run_name:
        out = output_root / run_name
    else:
        out = output_root / time.strftime("eval_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_eval_trainer(cfg: Dict[str, Any], deterministic: bool) -> TrainerSyncOnPolicy:
    env_cfg = dict(cfg["env"])
    env = MultiUAV2DEnv(**env_cfg)
    obs, _ = env.reset(seed=int(cfg.get("seed", 42)))
    aid = sorted(obs.keys())[0]
    high_obs = obs[aid]["high"]
    low_obs = obs[aid]["low"]
    obs_dim_high = int(high_obs.self_state.shape[0] + high_obs.goal_relative.shape[0] + high_obs.neighbor_summary.shape[0])
    obs_dim_low = int(low_obs.flat.shape[0])

    skills = build_default_skill_library(max_duration=int(cfg["skills"]["default_max_duration"]))
    n_skills = len(skills)

    high_policy = HighLevelPolicy(
        obs_dim=obs_dim_high,
        n_skills=n_skills,
        hidden_dim=int(cfg["model"]["high_hidden_dim"]),
    )
    low_policy = LowLevelQPPolicy(
        obs_dim=obs_dim_low,
        n_skills=n_skills,
        action_dim=int(cfg["model"]["action_dim"]),
        hidden_dim=int(cfg["model"]["low_hidden_dim"]),
    )

    action_limit = float(cfg["env"]["action_limit"])
    constraint_builder = ConstraintBuilder(
        d_min_agent=float(cfg["safety"]["d_min_agent"]),
        d_safe_obs=float(cfg["safety"]["d_safe_obs"]),
        u_min=[-action_limit, -action_limit],
        u_max=[action_limit, action_limit],
    )
    qp_solver = DifferentiableQPSolver(
        action_dim=int(cfg["model"]["action_dim"]),
        use_stub_if_unavailable=bool(cfg["qp"]["use_stub_if_unavailable"]),
        ecos_max_iters=int(cfg["qp"].get("ecos_max_iters", 500)),
        scs_max_iters=int(cfg["qp"].get("scs_max_iters", 10_000)),
        scs_eps=float(cfg["qp"].get("scs_eps", 1e-4)),
    )
    skill_params = dict(cfg["skills"]["params"])
    skill_params["action_limit"] = action_limit
    skill_params["cbf_u_max"] = action_limit
    runtime = SkillRuntimeManager(skills, default_ctx=skill_params)
    low_controller = LowLevelSafeController(
        low_policy=low_policy,
        constraint_builder=constraint_builder,
        qp_solver=qp_solver,
        skill_ref_weight=0.7,
        neighbor_perception_radius=float(cfg["env"]["neighbor_radius"]),
        obstacle_perception_range=float(cfg["env"]["lidar_range"]),
    )
    trainer = TrainerSyncOnPolicy(
        env=env,
        high_policy=high_policy,
        low_policy=low_policy,
        constraint_builder=constraint_builder,
        qp_solver=qp_solver,
        coordinator=SyncCoordinator(
            num_agents=int(cfg["env"]["n_agents"]),
            t_sync_max=int(cfg["synchronization"]["t_sync_max"]),
            mode=str(cfg["synchronization"].get("mode", "sync")),
        ),
        buffer=HierRolloutBuffer(),
        skill_runtime=runtime,
        low_level_controller=low_controller,
    )
    # Evaluation behavior only; no learning updates.
    trainer.hooks.eval_deterministic = bool(deterministic)
    return trainer


def _obs_batch(obs_map: Dict[int, Dict[str, Any]], agent_ids: List[int]) -> np.ndarray:
    rows = []
    for aid in agent_ids:
        h: AgentObsHigh = obs_map[aid]["high"]
        rows.append(np.concatenate([h.self_state, h.goal_relative, h.neighbor_summary], axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


def _sample_high_skills(
    trainer: TrainerSyncOnPolicy,
    obs_map: Dict[int, Dict[str, Any]],
    agent_ids: List[int],
    deterministic: bool,
) -> Dict[int, int]:
    batch = _obs_batch(obs_map, agent_ids)
    with torch.no_grad():
        out = trainer.high_policy.act(torch.as_tensor(batch, dtype=torch.float32), deterministic=deterministic)
    z = out["z"].detach().cpu().numpy()
    return {aid: int(z[idx]) for idx, aid in enumerate(agent_ids)}


def _activate_round_with_hover_fallback(
    trainer: TrainerSyncOnPolicy,
    sampled_skills: Dict[int, int],
    states: Dict[int, AgentState],
) -> Dict[int, int]:
    # Reuse trainer logic so behavior matches training/evaluation pipeline.
    return trainer.activate_round_skills(skill_map=sampled_skills, states=states)


def _evaluate_once(
    trainer: TrainerSyncOnPolicy,
    scene_seed: int,
    max_steps: int,
    deterministic: bool,
) -> Dict[str, Any]:
    obs, _ = trainer.env.reset(seed=int(scene_seed))
    agent_ids = sorted(obs.keys())
    trainer.skill_runtime.reset(agent_ids)
    trainer.coordinator.reset()

    reached_any = {aid: False for aid in agent_ids}
    unsafe_any = {aid: False for aid in agent_ids}

    sampled = _sample_high_skills(trainer, obs, agent_ids, deterministic=deterministic)
    states0 = {s.agent_id: s for s in trainer.env.get_agent_states()}
    active = _activate_round_with_hover_fallback(trainer, sampled, states0)

    steps = 0
    terminated = False
    truncated = False
    for _ in range(max_steps):
        steps += 1
        states = {s.agent_id: s for s in trainer.env.get_agent_states()}
        obs_low = {aid: obs[aid]["low"] for aid in agent_ids}
        actions, _ = trainer.compute_safe_actions(
            states=states,
            obs_low=obs_low,
            obstacles=trainer.env.get_obstacles(),
        )
        next_obs, _, terminated, truncated, info = trainer.env.step(actions)
        next_states = {s.agent_id: s for s in trainer.env.get_agent_states()}
        next_obs_low = {aid: next_obs[aid]["low"] for aid in agent_ids}
        skill_out = trainer.skill_runtime.step_all(
            states=next_states,
            obs_low=next_obs_low,
            executed_actions=actions,
        )
        beta = {aid: bool(skill_out[aid].beta) for aid in agent_ids}
        sync_res = trainer.coordinator.step(beta)
        forced_end = bool(terminated or truncated)
        switched_agents = set(sync_res.switch_agents)
        if forced_end:
            switched_agents = set(agent_ids)

        for aid in agent_ids:
            reached_any[aid] = bool(reached_any[aid] or bool(info.get("reach_flags", {}).get(aid, False)))
            unsafe_any[aid] = bool(unsafe_any[aid] or bool(info.get("unsafe_flags", {}).get(aid, False)))

        obs = next_obs
        if (len(switched_agents) > 0) and not (terminated or truncated):
            sampled = _sample_high_skills(trainer, obs, agent_ids, deterministic=deterministic)
            states_round = {s.agent_id: s for s in trainer.env.get_agent_states()}
            changed = _activate_round_with_hover_fallback(
                trainer,
                {aid: int(sampled[aid]) for aid in switched_agents},
                states_round,
            )
            active.update(changed)
        if terminated or truncated:
            break

    safe_reach_count = int(sum(1 for aid in agent_ids if reached_any[aid] and not unsafe_any[aid]))
    n_agents = int(len(agent_ids))
    success_round = bool(safe_reach_count == n_agents)
    return {
        "scene_seed": int(scene_seed),
        "steps": int(steps),
        "sim_seconds": float(steps * float(trainer.env.dt)),
        "safe_reach_count": safe_reach_count,
        "n_agents": n_agents,
        "safe_reach_ratio": float(safe_reach_count / max(1, n_agents)),
        "success_round": int(success_round),
        "terminated": int(bool(terminated)),
        "truncated": int(bool(truncated)),
        "active_skill_ids": json.dumps({int(k): int(v) for k, v in active.items()}, ensure_ascii=False),
    }


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else _resolve_latest_checkpoint()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
    cfg = _load_config(args=args, ckpt_payload=payload, ckpt_path=ckpt_path)

    # Enforce per-episode simulation duration (seconds -> horizon steps).
    dt = float(cfg["env"]["dt"])
    horizon_steps = max(1, int(math.ceil(float(args.seconds) / max(1e-8, dt))))
    cfg["env"] = dict(cfg["env"])
    cfg["env"]["horizon"] = horizon_steps

    out_dir = _make_output_dir(Path(args.output_root), args.run_name)
    trainer = _build_eval_trainer(cfg=cfg, deterministic=bool(args.deterministic))

    _load_state_dict_flexible(trainer.high_policy, payload["high_policy"], "high_policy")
    _load_state_dict_flexible(trainer.low_policy, payload["low_policy"], "low_policy")
    trainer.high_policy.eval()
    trainer.low_policy.eval()

    rows: List[Dict[str, Any]] = []
    total_safe_reach = 0
    total_agents = 0
    success_rounds = 0
    for ep in range(int(args.episodes)):
        row = _evaluate_once(
            trainer=trainer,
            scene_seed=int(args.seed) + ep,
            max_steps=horizon_steps,
            deterministic=bool(args.deterministic),
        )
        row["episode"] = int(ep)
        rows.append(row)

        total_safe_reach += int(row["safe_reach_count"])
        total_agents += int(row["n_agents"])
        success_rounds += int(row["success_round"])

        print(
            f"[episode {ep + 1}/{int(args.episodes)}] "
            f"safe_reach={row['safe_reach_count']}/{row['n_agents']} "
            f"success={row['success_round']} "
            f"steps={row['steps']} "
            f"sim_t={row['sim_seconds']:.2f}s"
        )

    overall_safe_reach_ratio = float(total_safe_reach / max(1, total_agents))
    summary = {
        "checkpoint": str(ckpt_path),
        "episodes": int(args.episodes),
        "seconds_per_episode": float(args.seconds),
        "horizon_steps": int(horizon_steps),
        "seed_base": int(args.seed),
        "deterministic": bool(args.deterministic),
        "total_safe_reach": int(total_safe_reach),
        "total_agents": int(total_agents),
        "overall_safe_reach_ratio": overall_safe_reach_ratio,
        "success_rounds": int(success_rounds),
        "success_round_rate": float(success_rounds / max(1, int(args.episodes))),
        "output_dir": str(out_dir),
    }

    _write_rows_csv(out_dir / "episode_results.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("==== Evaluation Summary ====")
    print(f"checkpoint: {ckpt_path}")
    print(f"episodes: {int(args.episodes)}")
    print(f"per-episode duration: {float(args.seconds):.2f}s ({horizon_steps} steps @ dt={dt:.4f})")
    print(f"safe_reach_total: {total_safe_reach}/{total_agents} ({overall_safe_reach_ratio:.4f})")
    print(f"success_rounds: {success_rounds}/{int(args.episodes)} ({summary['success_round_rate']:.4f})")
    print(f"results saved to: {out_dir}")


if __name__ == "__main__":
    main()
