from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime entrypoint
    raise RuntimeError("PyTorch is required to run multi-agent high-level-only training") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime entrypoint
    raise RuntimeError("PyYAML is required to load config") from exc

from hmarl_cbf.buffer import HierRolloutBuffer
from hmarl_cbf.control import SyncCoordinator
from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.high_level import MAPPOConfig, OnPolicyMAPPO
from hmarl_cbf.policies import HighLevelPolicy
from hmarl_cbf.skills import SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.train.eval_high_level_multiagent_skills_only import _default_states, _parse_obstacles, _parse_states
from hmarl_cbf.train.run_sync_onpolicy import _make_run_dir, _seed_all, _write_history_csv
from hmarl_cbf.types import AgentState, HighOptionTransition


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train only the high-level skill policy on a fixed multi-UAV task using skill semantics as the executor."
    )
    parser.add_argument("--config", type=str, default="configs/hmarl_cbf/default_async_onpolicy_gcbfplus.yaml")
    parser.add_argument("--output-root", type=str, default="artifacts/hmarl_cbf_paper_async")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--total-iterations", type=int, default=200)
    parser.add_argument("--episodes-per-iter", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--invalid-skill-penalty", type=float, default=-1.0)
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--states-json", type=str, default="")
    parser.add_argument("--obstacles-json", type=str, default="[]")
    parser.add_argument("--sync-mode", type=str, default="", choices=["", "sync", "async"])
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args()


def _obs_high_vec(obs_high: Any) -> np.ndarray:
    return np.concatenate([obs_high.self_state, obs_high.goal_relative, obs_high.neighbor_summary], axis=0).astype(np.float32)


def _fixed_env(cfg: Dict[str, Any], args: argparse.Namespace, seed: int) -> tuple[MultiUAV2DEnv, Dict[int, Dict[str, Any]], int]:
    env_cfg = dict(cfg["env"])
    temp_env = MultiUAV2DEnv(**env_cfg)
    states = _parse_states(str(args.states_json), agent_radius=float(temp_env.agent_radius))
    obstacles = _parse_obstacles(str(args.obstacles_json))
    env_cfg["n_agents"] = len(states)
    env_cfg["n_obstacles"] = len(obstacles)
    env = MultiUAV2DEnv(**env_cfg)
    obs, _ = env.reset(seed=int(seed), options={"states": states, "obstacles": obstacles})
    return env, obs, len(states)


def _sample_option(policy: HighLevelPolicy, obs_high: Any, deterministic: bool) -> Dict[str, float | int]:
    obs_vec = torch.as_tensor(_obs_high_vec(obs_high), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = policy.act(obs_vec, deterministic=deterministic)
    return {
        "skill_id": int(out["z"].detach().cpu().item()),
        "logp": float(out["logp"].detach().cpu().item()),
        "value": float(out["value"].detach().cpu().item()),
    }


def _list_valid_skill_ids(runtime: SkillRuntimeManager, agent_id: int, state: AgentState) -> List[int]:
    valid: List[int] = []
    for skill_id, skill in runtime.skill_by_id.items():
        ctx = runtime._prepare_agent_ctx(agent_id=agent_id, skill=skill, state=state)  # type: ignore[attr-defined]
        if skill.initiation_set_fn(state, ctx):
            valid.append(int(skill_id))
    return sorted(valid)


def _sample_valid_option(
    policy: HighLevelPolicy,
    obs_high: Any,
    valid_skill_ids: List[int],
    deterministic: bool,
) -> Dict[str, float | int]:
    if not valid_skill_ids:
        raise ValueError("valid_skill_ids must be non-empty")
    obs_vec = torch.as_tensor(_obs_high_vec(obs_high), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, value = policy.forward(obs_vec)
        masked_logits = torch.full_like(logits, -1e9)
        masked_logits[:, valid_skill_ids] = logits[:, valid_skill_ids]
        dist = torch.distributions.Categorical(logits=masked_logits)
        if deterministic:
            z = torch.argmax(masked_logits, dim=-1)
        else:
            z = dist.sample()
        logp = dist.log_prob(z)
    return {
        "skill_id": int(z.detach().cpu().item()),
        "logp": float(logp.detach().cpu().item()),
        "value": float(value.detach().cpu().item()),
    }


def _set_high_entropy_coef(updater: OnPolicyMAPPO, train_cfg: Dict[str, Any], itr: int, total_iterations: int) -> float:
    default_coef = float(updater.config.entropy_coef)
    start = float(train_cfg.get("high_entropy_coef_start", default_coef))
    end = float(train_cfg.get("high_entropy_coef_end", start))
    decay_iters = max(1, int(train_cfg.get("high_entropy_decay_iters", total_iterations)))
    if decay_iters <= 1:
        coef = end
    else:
        alpha = min(1.0, max(0.0, float(itr - 1) / float(decay_iters - 1)))
        coef = start + (end - start) * alpha
    updater.config.entropy_coef = float(coef)
    return float(coef)


def _finalize_open_options(
    *,
    buffer: HierRolloutBuffer,
    agent_ids: List[int],
    t_end: int,
    round_return_ext: Dict[int, float],
    done: bool,
) -> List[HighOptionTransition]:
    closed: List[HighOptionTransition] = []
    for aid in agent_ids:
        if not buffer.has_open_high_option(aid):
            continue
        closed.append(
            buffer.close_high_option(
                agent_id=aid,
                t_end=t_end,
                return_ext=float(round_return_ext[aid]),
                done=done,
                sync_switch=False,
            )
        )
    return closed


def _apply_team_success_bonus(
    *,
    transitions: List[HighOptionTransition],
    agent_ids: List[int],
    episode_return: Dict[int, float],
    bonus: float,
) -> None:
    if bonus == 0.0:
        return
    latest_idx_by_agent: Dict[int, int] = {}
    for idx, tr in enumerate(transitions):
        latest_idx_by_agent[tr.agent_id] = idx
    for aid in agent_ids:
        if aid in latest_idx_by_agent:
            transitions[latest_idx_by_agent[aid]].return_ext += float(bonus)
            episode_return[aid] += float(bonus)


def _run_episode(
    *,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    policy: HighLevelPolicy,
    skills,
    seed: int,
    deterministic: bool,
    invalid_skill_penalty: float,
    mode: str,
) -> Dict[str, Any]:
    env, obs, _ = _fixed_env(cfg, args, seed=seed)
    agent_ids = sorted(obs.keys())
    states = {s.agent_id: s for s in env.get_agent_states()}
    runtime = SkillRuntimeManager(
        skills,
        default_ctx={
            **dict(cfg["skills"]["params"]),
            "action_limit": float(cfg["env"]["action_limit"]),
            "cbf_u_max": float(cfg["env"]["action_limit"]),
            "dt": float(cfg["env"]["dt"]),
        },
    )
    runtime.reset(agent_ids)
    coordinator = SyncCoordinator(
        num_agents=len(agent_ids),
        t_sync_max=int(cfg["synchronization"]["t_sync_max"]),
        mode=mode,
    )
    coordinator.reset()
    buffer = HierRolloutBuffer()
    gamma_high = float(cfg["train"]["gamma_high"])
    lam_high = float(cfg["train"]["lam_high"])
    round_return_ext = {aid: 0.0 for aid in agent_ids}
    round_discount = {aid: 1.0 for aid in agent_ids}
    chosen_skills_by_agent: Dict[int, List[Dict[str, Any]]] = {aid: [] for aid in agent_ids}

    for aid in agent_ids:
        valid_skill_ids = _list_valid_skill_ids(runtime, agent_id=aid, state=states[aid])
        sample = _sample_valid_option(policy, obs[aid]["high"], valid_skill_ids, deterministic=deterministic)
        skill_id = int(sample["skill_id"])
        try:
            runtime.activate_skill(agent_id=aid, skill_id=skill_id, state=states[aid])
        except ValueError:
            buffer.add_high_option(
                HighOptionTransition(
                    k=int(coordinator.option_k[aid]),
                    agent_id=aid,
                    t_start=0,
                    t_end=0,
                    obs_high=obs[aid]["high"],
                    skill_id=skill_id,
                    logp=float(sample["logp"]),
                    value=float(sample["value"]),
                    return_ext=float(invalid_skill_penalty),
                    done=True,
                    info={"invalid_skill": True},
                )
            )
            continue
        buffer.start_high_option(
            k=int(coordinator.option_k[aid]),
            agent_id=aid,
            t_start=0,
            obs_high=obs[aid]["high"],
            skill_id=skill_id,
            logp=float(sample["logp"]),
            value=float(sample["value"]),
        )
        chosen_skills_by_agent[aid].append({"t": 0, "skill_id": skill_id, "skill_name": skills[skill_id].name})

    episode_return = {aid: 0.0 for aid in agent_ids}
    terminated = False
    truncated = False
    info: Dict[str, Any] = {"reach_flags": {}, "unsafe_flags": {}}
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(cfg["env"]["horizon"])

    for _ in range(max_steps):
        states = {s.agent_id: s for s in env.get_agent_states()}
        actions: Dict[int, np.ndarray] = {}
        for aid in agent_ids:
            if buffer.has_open_high_option(aid):
                target = runtime.control_target(agent_id=aid, state=states[aid], obs_low=obs[aid]["low"])
                actions[aid] = np.asarray(target["u_ref_skill"], dtype=np.float32).reshape(2)
            else:
                actions[aid] = np.zeros(2, dtype=np.float32)

        next_obs, rewards, terminated, truncated, info = env.step(actions)
        next_states = {s.agent_id: s for s in env.get_agent_states()}
        active_states = {aid: next_states[aid] for aid in agent_ids if buffer.has_open_high_option(aid)}
        active_obs = {aid: next_obs[aid]["low"] for aid in agent_ids if buffer.has_open_high_option(aid)}
        active_actions = {aid: actions[aid] for aid in agent_ids if buffer.has_open_high_option(aid)}
        skill_out = runtime.step_all(states=active_states, obs_low=active_obs, executed_actions=active_actions) if active_states else {}
        beta = {aid: bool(skill_out[aid].beta) if aid in skill_out else False for aid in agent_ids}
        sync_res = coordinator.step(beta)

        for aid in agent_ids:
            reward = float(rewards[aid])
            episode_return[aid] += reward
            if not buffer.has_open_high_option(aid):
                continue
            round_return_ext[aid] += round_discount[aid] * reward
            round_discount[aid] *= gamma_high

        if sync_res.sync_switch:
            for aid in sync_res.switch_agents:
                if buffer.has_open_high_option(aid):
                    buffer.close_high_option(
                        agent_id=aid,
                        t_end=int(coordinator.t),
                        return_ext=float(round_return_ext[aid]),
                        done=bool(terminated or truncated),
                        sync_switch=True,
                    )
                    round_return_ext[aid] = 0.0
                    round_discount[aid] = 1.0

                if terminated or truncated:
                    continue

                valid_skill_ids = _list_valid_skill_ids(runtime, agent_id=aid, state=next_states[aid])
                sample = _sample_valid_option(
                    policy,
                    next_obs[aid]["high"],
                    valid_skill_ids,
                    deterministic=deterministic,
                )
                skill_id = int(sample["skill_id"])
                try:
                    runtime.activate_skill(agent_id=aid, skill_id=skill_id, state=next_states[aid])
                except ValueError:
                    buffer.add_high_option(
                        HighOptionTransition(
                            k=int(coordinator.option_k[aid]),
                            agent_id=aid,
                            t_start=int(coordinator.t),
                            t_end=int(coordinator.t),
                            obs_high=next_obs[aid]["high"],
                            skill_id=skill_id,
                            logp=float(sample["logp"]),
                            value=float(sample["value"]),
                            return_ext=float(invalid_skill_penalty),
                            done=False,
                            info={"invalid_skill": True},
                            sync_switch=True,
                        )
                    )
                    continue

                buffer.start_high_option(
                    k=int(coordinator.option_k[aid]),
                    agent_id=aid,
                    t_start=int(coordinator.t),
                    obs_high=next_obs[aid]["high"],
                    skill_id=skill_id,
                    logp=float(sample["logp"]),
                    value=float(sample["value"]),
                )
                chosen_skills_by_agent[aid].append(
                    {"t": int(coordinator.t), "skill_id": skill_id, "skill_name": skills[skill_id].name}
                )

        obs = next_obs
        if terminated or truncated:
            break

    _ = _finalize_open_options(
        buffer=buffer,
        agent_ids=agent_ids,
        t_end=int(coordinator.t),
        round_return_ext=round_return_ext,
        done=True,
    )
    team_success = all(
        bool(info.get("reach_flags", {}).get(aid, False)) and not bool(info.get("unsafe_flags", {}).get(aid, False))
        for aid in agent_ids
    )
    if team_success:
        _apply_team_success_bonus(
            transitions=buffer.high_options,
            agent_ids=agent_ids,
            episode_return=episode_return,
            bonus=float(cfg.get("train", {}).get("team_success_bonus", 0.0)),
        )
    buffer.compute_high_advantages(
        gamma=gamma_high,
        lam=lam_high,
        use_gae=True,
        bootstrap_value_by_agent={},
    )
    per_agent_success = {
        aid: bool(info.get("reach_flags", {}).get(aid, False)) and not bool(info.get("unsafe_flags", {}).get(aid, False))
        for aid in agent_ids
    }
    return {
        "transitions": list(buffer.high_options),
        "episode_return_mean": float(sum(episode_return.values()) / max(1, len(agent_ids))),
        "team_success": bool(team_success),
        "per_agent_success_mean": float(sum(1.0 if per_agent_success[aid] else 0.0 for aid in agent_ids) / max(1, len(agent_ids))),
        "collision_rate_mean": float(sum(1.0 if bool(info.get("unsafe_flags", {}).get(aid, False)) else 0.0 for aid in agent_ids) / max(1, len(agent_ids))),
        "chosen_skills_by_agent": chosen_skills_by_agent,
    }


def _evaluate_policy(
    *,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    policy: HighLevelPolicy,
    skills,
    base_seed: int,
    episodes: int,
    deterministic: bool,
    invalid_skill_penalty: float,
    mode: str,
) -> Dict[str, Any]:
    returns: List[float] = []
    team_success_vals: List[float] = []
    per_agent_success_vals: List[float] = []
    collision_vals: List[float] = []
    for ep in range(episodes):
        rollout = _run_episode(
            cfg=cfg,
            args=args,
            policy=policy,
            skills=skills,
            seed=base_seed + ep,
            deterministic=deterministic,
            invalid_skill_penalty=invalid_skill_penalty,
            mode=mode,
        )
        returns.append(float(rollout["episode_return_mean"]))
        team_success_vals.append(1.0 if bool(rollout["team_success"]) else 0.0)
        per_agent_success_vals.append(float(rollout["per_agent_success_mean"]))
        collision_vals.append(float(rollout["collision_rate_mean"]))
    return {
        "eval_episode_return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval_team_success_rate": float(np.mean(team_success_vals)) if team_success_vals else 0.0,
        "eval_agent_success_rate": float(np.mean(per_agent_success_vals)) if per_agent_success_vals else 0.0,
        "eval_collision_rate": float(np.mean(collision_vals)) if collision_vals else 0.0,
    }


def main() -> None:
    args = _parse_args()
    cfg: Dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42) if args.seed < 0 else args.seed)
    _seed_all(seed)

    _, obs, _ = _fixed_env(cfg, args, seed=seed)
    agent_id = sorted(obs.keys())[0]
    obs_high = obs[agent_id]["high"]
    obs_dim_high = int(obs_high.self_state.shape[0] + obs_high.goal_relative.shape[0] + obs_high.neighbor_summary.shape[0])
    skills = build_default_skill_library(max_duration=int(cfg["skills"]["default_max_duration"]))
    high_policy = HighLevelPolicy(
        obs_dim=obs_dim_high,
        n_skills=len(skills),
        hidden_dim=int(cfg["model"]["high_hidden_dim"]),
    )
    high_opt = torch.optim.Adam(high_policy.parameters(), lr=float(args.learning_rate))
    high_updater = OnPolicyMAPPO(
        policy=high_policy,
        optimizer=high_opt,
        config=MAPPOConfig(**cfg["high_level_mappo"]),
    )

    mode = str(args.sync_mode).strip().lower() or str(cfg["synchronization"].get("mode", "async")).strip().lower()
    output_root = Path(args.output_root)
    run_dir = _make_run_dir(output_root=output_root, run_name=args.run_name or "high_level_multiagent_only")
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    total_iterations = int(args.total_iterations)
    episodes_per_iter = max(1, int(args.episodes_per_iter))
    eval_interval = max(1, int(args.eval_interval))
    invalid_skill_penalty = float(args.invalid_skill_penalty)
    history: List[Dict[str, float]] = []
    last_eval: Dict[str, Any] = {}

    for itr in range(1, total_iterations + 1):
        high_entropy_coef = _set_high_entropy_coef(high_updater, cfg["train"], itr, total_iterations)
        episode_returns: List[float] = []
        team_success_vals: List[float] = []
        per_agent_success_vals: List[float] = []
        collision_vals: List[float] = []
        transitions: List[HighOptionTransition] = []

        for ep in range(episodes_per_iter):
            rollout = _run_episode(
                cfg=cfg,
                args=args,
                policy=high_policy,
                skills=skills,
                seed=seed + itr * 1000 + ep,
                deterministic=False,
                invalid_skill_penalty=invalid_skill_penalty,
                mode=mode,
            )
            episode_returns.append(float(rollout["episode_return_mean"]))
            team_success_vals.append(1.0 if bool(rollout["team_success"]) else 0.0)
            per_agent_success_vals.append(float(rollout["per_agent_success_mean"]))
            collision_vals.append(float(rollout["collision_rate_mean"]))
            transitions.extend(rollout["transitions"])

        update_stats = high_updater.update(transitions)
        row: Dict[str, float] = {
            "iteration": float(itr),
            "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "train_team_success_rate": float(np.mean(team_success_vals)) if team_success_vals else 0.0,
            "train_agent_success_rate": float(np.mean(per_agent_success_vals)) if per_agent_success_vals else 0.0,
            "train_collision_rate": float(np.mean(collision_vals)) if collision_vals else 0.0,
            "high_samples": float(update_stats.get("n_samples", 0.0)),
            "loss_high_total": float(update_stats.get("loss_total", 0.0)),
            "loss_high_actor": float(update_stats.get("loss_actor", 0.0)),
            "loss_high_value": float(update_stats.get("loss_value", 0.0)),
            "high_entropy": float(update_stats.get("entropy", 0.0)),
            "high_approx_kl": float(update_stats.get("approx_kl", 0.0)),
            "high_clip_frac": float(update_stats.get("clip_frac", 0.0)),
            "high_entropy_coef": float(high_entropy_coef),
        }

        if itr == 1 or itr % eval_interval == 0 or itr == total_iterations:
            eval_stats = _evaluate_policy(
                cfg=cfg,
                args=args,
                policy=high_policy,
                skills=skills,
                base_seed=seed + 100_000 + itr * 10,
                episodes=max(1, int(args.eval_episodes)),
                deterministic=bool(args.deterministic_eval),
                invalid_skill_penalty=invalid_skill_penalty,
                mode=mode,
            )
            last_eval = eval_stats
            row.update({k: float(v) for k, v in eval_stats.items()})
            print(
                f"[iter {itr}/{total_iterations}] "
                f"train_ret={row['episode_return_mean']:.4f} "
                f"train_team={row['train_team_success_rate']:.4f} "
                f"eval_team={row['eval_team_success_rate']:.4f} "
                f"eval_agent={row['eval_agent_success_rate']:.4f} "
                f"eval_coll={row['eval_collision_rate']:.4f} "
                f"entropy={row['high_entropy']:.4f} "
                f"ent_coef={row['high_entropy_coef']:.4f}"
            )
        else:
            print(
                f"[iter {itr}/{total_iterations}] "
                f"train_ret={row['episode_return_mean']:.4f} "
                f"train_team={row['train_team_success_rate']:.4f} "
                f"entropy={row['high_entropy']:.4f} "
                f"ent_coef={row['high_entropy_coef']:.4f}"
            )
        history.append(row)

    final_eval = _evaluate_policy(
        cfg=cfg,
        args=args,
        policy=high_policy,
        skills=skills,
        base_seed=seed + 900_000,
        episodes=max(1, int(args.eval_episodes)),
        deterministic=True,
        invalid_skill_penalty=invalid_skill_penalty,
        mode=mode,
    )

    torch.save(
        {
            "seed": seed,
            "config": cfg,
            "high_policy": high_policy.state_dict(),
            "high_optimizer": high_opt.state_dict(),
            "mode": "high_level_only_multiagent",
        },
        run_dir / "checkpoints" / "last.pt",
    )
    _write_history_csv(run_dir / "train_history.csv", history)
    summary = {
        "seed": seed,
        "mode": "high_level_only_multiagent",
        "sync_mode": str(mode),
        "states_json": str(args.states_json) if str(args.states_json).strip() else json.dumps(_default_states(), ensure_ascii=False),
        "obstacles_json": str(args.obstacles_json),
        "total_iterations": total_iterations,
        "episodes_per_iter": episodes_per_iter,
        "invalid_skill_penalty": invalid_skill_penalty,
        "team_success_bonus": float(cfg.get("train", {}).get("team_success_bonus", 0.0)),
        "last_eval_during_train": last_eval,
        "final_eval": final_eval,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "LATEST_RUN").write_text(str(run_dir), encoding="utf-8")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
