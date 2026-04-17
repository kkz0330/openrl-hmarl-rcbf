from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime entrypoint
    raise RuntimeError("PyTorch is required to run high-level-only training") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime entrypoint
    raise RuntimeError("PyYAML is required to load config") from exc

from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.eval import EpisodeTrace, TrajectoryRenderer
from hmarl_cbf.high_level import MAPPOConfig, OnPolicyMAPPO
from hmarl_cbf.policies import HighLevelPolicy
from hmarl_cbf.skills import SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.train.run_sync_onpolicy import _make_run_dir, _seed_all, _write_history_csv
from hmarl_cbf.types import AgentState, HighOptionTransition


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train only the high-level skill policy on a fixed single-UAV task using skill semantics as the executor."
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
    parser.add_argument("--final-render-gif", action="store_true")
    parser.add_argument("--final-render-png", action="store_true")
    parser.add_argument("--start-x", type=float, default=-4.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--goal-x", type=float, default=4.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--obstacle-x", type=float, default=0.0)
    parser.add_argument("--obstacle-y", type=float, default=0.0)
    parser.add_argument("--obstacle-r", type=float, default=1.0)
    parser.add_argument("--init-vx", type=float, default=0.0)
    parser.add_argument("--init-vy", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args()


def _obs_high_vec(obs_high: Any) -> np.ndarray:
    return np.concatenate([obs_high.self_state, obs_high.goal_relative, obs_high.neighbor_summary], axis=0).astype(np.float32)


def _segment_clear_of_circle(start: np.ndarray, goal: np.ndarray, center: np.ndarray, radius: float) -> bool:
    seg = np.asarray(goal - start, dtype=np.float32).reshape(2)
    seg_norm_sq = float(np.dot(seg, seg))
    if seg_norm_sq <= 1e-8:
        return float(np.linalg.norm(start - center)) > radius
    t = float(np.dot(center - start, seg) / seg_norm_sq)
    t = min(1.0, max(0.0, t))
    closest = start + t * seg
    return float(np.linalg.norm(closest - center)) > radius


def _goal_visibility_margin(state: AgentState, obstacles: List[Dict[str, Any]]) -> float:
    start = np.asarray(state.position, dtype=np.float32).reshape(2)
    goal = np.asarray(state.goal, dtype=np.float32).reshape(2)
    clearance = float(state.radius)
    if not obstacles:
        return float("inf")
    min_margin = float("inf")
    for obstacle in obstacles:
        center = np.asarray(obstacle["center"], dtype=np.float32).reshape(2)
        radius = float(obstacle["radius"]) + clearance
        seg = np.asarray(goal - start, dtype=np.float32).reshape(2)
        seg_norm_sq = float(np.dot(seg, seg))
        if seg_norm_sq <= 1e-8:
            margin = float(np.linalg.norm(start - center) - radius)
        else:
            t = float(np.dot(center - start, seg) / seg_norm_sq)
            t = min(1.0, max(0.0, t))
            closest = start + t * seg
            margin = float(np.linalg.norm(closest - center) - radius)
        min_margin = min(min_margin, margin)
    return float(min_margin)


def _goal_visibility_reward(
    prev_margin: float,
    curr_margin: float,
    visible_bonus_scale: float,
    improvement_scale: float,
) -> float:
    reward = 0.0
    if np.isfinite(prev_margin) and np.isfinite(curr_margin):
        reward += float(improvement_scale) * float(curr_margin - prev_margin)
    if curr_margin > 0.0:
        reward += float(visible_bonus_scale)
    return float(reward)


def _goal_visibility_bonus(state: AgentState, obstacles: List[Dict[str, Any]], reward_scale: float) -> float:
    if reward_scale <= 0.0:
        return 0.0
    return float(reward_scale) if _goal_visibility_margin(state, obstacles) > 0.0 else 0.0


def _goal_heading_alignment_penalty(state: AgentState, penalty_scale: float) -> float:
    if penalty_scale <= 0.0:
        return 0.0
    velocity = np.asarray(state.velocity, dtype=np.float32).reshape(2)
    goal_vec = np.asarray(state.goal - state.position, dtype=np.float32).reshape(2)
    vel_norm = float(np.linalg.norm(velocity))
    goal_norm = float(np.linalg.norm(goal_vec))
    if vel_norm <= 1e-6 or goal_norm <= 1e-6:
        return 0.0
    vel_dir = velocity / vel_norm
    goal_dir = goal_vec / goal_norm
    cos_theta = float(np.clip(np.dot(vel_dir, goal_dir), -1.0, 1.0))
    # 0 when aligned, 1 when orthogonal, 2 when opposite.
    return float(penalty_scale) * (1.0 - cos_theta)


def _fixed_env(cfg: Dict[str, Any], args: argparse.Namespace, seed: int) -> tuple[MultiUAV2DEnv, Dict[int, Dict[str, Any]]]:
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
    obs, _ = env.reset(seed=int(seed), options={"states": states, "obstacles": obstacles})
    return env, obs


def _sample_option(policy: HighLevelPolicy, obs_high: Any, deterministic: bool) -> Dict[str, float | int]:
    obs_vec = torch.as_tensor(_obs_high_vec(obs_high), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = policy.act(obs_vec, deterministic=deterministic)
    return {
        "skill_id": int(out["z"].detach().cpu().item()),
        "logp": float(out["logp"].detach().cpu().item()),
        "value": float(out["value"].detach().cpu().item()),
    }


def _try_activate_skill(runtime: SkillRuntimeManager, agent_id: int, skill_id: int, state: AgentState) -> bool:
    try:
        runtime.activate_skill(agent_id=agent_id, skill_id=skill_id, state=state)
    except ValueError:
        return False
    return True


def _discount_high_option_returns(transitions: List[HighOptionTransition], gamma: float) -> None:
    running = 0.0
    for transition in reversed(transitions):
        duration = max(1, int(transition.t_end - transition.t_start))
        if transition.done:
            running = float(transition.return_ext)
        else:
            running = float(transition.return_ext) + float(gamma**duration) * running
        transition.value_target = float(running)
        transition.advantage = float(running - transition.value)


def _run_episode(
    *,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    policy: HighLevelPolicy,
    skills,
    seed: int,
    deterministic: bool,
    invalid_skill_penalty: float,
    render_path: Path | None = None,
) -> Dict[str, Any]:
    env, obs = _fixed_env(cfg, args, seed=seed)
    agent_id = 0
    states = {s.agent_id: s for s in env.get_agent_states()}
    runtime = SkillRuntimeManager(
        skills,
        default_ctx={
            **dict(cfg["skills"]["params"]),
            "action_limit": float(cfg["env"]["action_limit"]),
            "cbf_u_max": float(cfg["env"]["action_limit"]),
            "dt": float(cfg["env"]["dt"]),
            "robust_cbf": bool(cfg.get("safety", {}).get("robust_cbf", False)),
            "disturbance_accel_max": float(cfg["env"].get("disturbance_accel_max", 0.0)),
            "relative_disturbance_accel_max": float(cfg.get("safety", {}).get("relative_disturbance_accel_max", 0.0)),
        },
    )
    runtime.reset([agent_id])
    visibility_reward = float(cfg.get("train", {}).get("goal_visibility_reward", 0.0))
    visibility_improvement_reward = float(cfg.get("train", {}).get("goal_visibility_improvement_reward", 0.0))
    goal_heading_alignment_penalty = float(cfg.get("train", {}).get("goal_heading_alignment_penalty", 0.0))
    prev_visibility_margin = _goal_visibility_margin(states[agent_id], env.get_obstacles())

    skill_names = [skill.name for skill in skills]
    option_index = 0
    episode_return = 0.0
    transitions: List[HighOptionTransition] = []
    chosen_skills: List[Dict[str, Any]] = []
    trace_positions: List[np.ndarray] = []
    trace_unsafe: List[np.ndarray] = []
    frame_labels: List[str] = []
    info: Dict[str, Any] = {"reach_flags": {agent_id: False}, "unsafe_flags": {agent_id: False}}

    option_sample = _sample_option(policy, obs[agent_id]["high"], deterministic=deterministic)
    current_skill_id = int(option_sample["skill_id"])
    current_logp = float(option_sample["logp"])
    current_value = float(option_sample["value"])
    current_obs_high = obs[agent_id]["high"]
    current_t_start = 0
    current_return = 0.0
    chosen_skills.append({"t": 0, "skill_id": current_skill_id, "skill_name": skill_names[current_skill_id]})

    if not _try_activate_skill(runtime, agent_id=agent_id, skill_id=current_skill_id, state=states[agent_id]):
        transitions.append(
            HighOptionTransition(
                k=option_index,
                agent_id=agent_id,
                t_start=0,
                t_end=0,
                obs_high=current_obs_high,
                skill_id=current_skill_id,
                logp=current_logp,
                value=current_value,
                return_ext=float(invalid_skill_penalty),
                done=True,
                info={"invalid_skill": True},
            )
        )
        _discount_high_option_returns(transitions, gamma=float(cfg["train"]["gamma_high"]))
        return {
            "transitions": transitions,
            "episode_return": float(invalid_skill_penalty),
            "reached_goal": False,
            "unsafe": False,
            "chosen_skills": chosen_skills,
            "steps": 0,
            "render_path": "",
        }

    terminated = False
    truncated = False
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(cfg["env"]["horizon"])
    for t in range(max_steps):
        states = {s.agent_id: s for s in env.get_agent_states()}
        obs_low = obs[agent_id]["low"]
        target = runtime.control_target(agent_id=agent_id, state=states[agent_id], obs_low=obs_low)
        action = np.asarray(target["u_ref_skill"], dtype=np.float32).reshape(2)

        next_obs, rewards, terminated, truncated, info = env.step({agent_id: action})
        next_states = {s.agent_id: s for s in env.get_agent_states()}
        step_out = runtime.step(
            agent_id=agent_id,
            state=next_states[agent_id],
            obs_low=next_obs[agent_id]["low"],
            executed_action=action,
        )

        reward = float(rewards[agent_id])
        current_visibility_margin = _goal_visibility_margin(next_states[agent_id], env.get_obstacles())
        reward += _goal_visibility_bonus(
            state=next_states[agent_id],
            obstacles=env.get_obstacles(),
            reward_scale=visibility_reward,
        )
        reward += _goal_visibility_reward(
            prev_margin=prev_visibility_margin,
            curr_margin=current_visibility_margin,
            visible_bonus_scale=0.0,
            improvement_scale=visibility_improvement_reward,
        )
        if current_visibility_margin > 0.0:
            reward -= _goal_heading_alignment_penalty(
                state=next_states[agent_id],
                penalty_scale=goal_heading_alignment_penalty,
            )
        prev_visibility_margin = current_visibility_margin
        current_return += reward
        episode_return += reward
        trace_positions.append(np.asarray(next_states[agent_id].position, dtype=np.float32).reshape(1, 2))
        trace_unsafe.append(np.asarray([bool(info.get("unsafe_flags", {}).get(agent_id, False))], dtype=bool))
        wind_accel = np.asarray(info.get("wind_accel", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
        frame_labels.append(
            f"t={t} skill={skill_names[current_skill_id]} reward={reward:.3f}  "
            f"wind=({wind_accel[0]:+0.2f}, {wind_accel[1]:+0.2f})  |w|={float(np.linalg.norm(wind_accel)):.2f}"
        )
        obs = next_obs

        should_close_option = bool(step_out.beta or terminated or truncated)
        if not should_close_option:
            continue

        done = bool(terminated or truncated)
        transitions.append(
            HighOptionTransition(
                k=option_index,
                agent_id=agent_id,
                t_start=current_t_start,
                t_end=t + 1,
                obs_high=current_obs_high,
                skill_id=current_skill_id,
                logp=current_logp,
                value=current_value,
                return_ext=float(current_return),
                done=done,
                info={"duration": int(t + 1 - current_t_start)},
            )
        )
        option_index += 1

        if done:
            break

        current_t_start = t + 1
        current_return = 0.0
        current_obs_high = obs[agent_id]["high"]
        option_sample = _sample_option(policy, current_obs_high, deterministic=deterministic)
        current_skill_id = int(option_sample["skill_id"])
        current_logp = float(option_sample["logp"])
        current_value = float(option_sample["value"])
        chosen_skills.append({"t": t + 1, "skill_id": current_skill_id, "skill_name": skill_names[current_skill_id]})
        if not _try_activate_skill(runtime, agent_id=agent_id, skill_id=current_skill_id, state=next_states[agent_id]):
            transitions.append(
                HighOptionTransition(
                    k=option_index,
                    agent_id=agent_id,
                    t_start=current_t_start,
                    t_end=current_t_start,
                    obs_high=current_obs_high,
                    skill_id=current_skill_id,
                    logp=current_logp,
                    value=current_value,
                    return_ext=float(invalid_skill_penalty),
                    done=True,
                    info={"invalid_skill": True},
                )
            )
            episode_return += float(invalid_skill_penalty)
            break

    _discount_high_option_returns(transitions, gamma=float(cfg["train"]["gamma_high"]))

    media_path = ""
    if render_path is not None and trace_positions:
        renderer = TrajectoryRenderer(world_size=float(getattr(env, "world_size", 10.0)))
        trace = EpisodeTrace(
            positions=np.stack(trace_positions, axis=0),
            goals=np.stack([np.asarray(s.goal, dtype=np.float32).reshape(2) for s in env.get_agent_states()], axis=0),
            obstacles=env.get_obstacles(),
            unsafe_flags=np.stack(trace_unsafe, axis=0),
            frame_labels=frame_labels,
        )
        if render_path.suffix.lower() == ".png":
            media_path = renderer.render_static(trace, render_path)
        else:
            media_path = renderer.render_gif(trace, render_path, fps=8)

    return {
        "transitions": transitions,
        "episode_return": float(episode_return),
        "reached_goal": bool(info.get("reach_flags", {}).get(agent_id, False)),
        "unsafe": bool(info.get("unsafe_flags", {}).get(agent_id, False)),
        "chosen_skills": chosen_skills,
        "steps": int(len(trace_positions)),
        "render_path": media_path,
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
    render_dir: Path | None = None,
    render_gif: bool = True,
) -> Dict[str, Any]:
    returns: List[float] = []
    success_flags: List[float] = []
    unsafe_flags: List[float] = []
    step_counts: List[float] = []
    media_paths: List[str] = []

    for ep in range(episodes):
        render_path = None
        if render_dir is not None and ep == 0:
            suffix = ".gif" if render_gif else ".png"
            render_path = render_dir / f"episode_000{suffix}"
        rollout = _run_episode(
            cfg=cfg,
            args=args,
            policy=policy,
            skills=skills,
            seed=base_seed + ep,
            deterministic=deterministic,
            invalid_skill_penalty=invalid_skill_penalty,
            render_path=render_path,
        )
        returns.append(float(rollout["episode_return"]))
        unsafe = bool(rollout["unsafe"])
        reached = bool(rollout["reached_goal"])
        success_flags.append(1.0 if (reached and not unsafe) else 0.0)
        unsafe_flags.append(1.0 if unsafe else 0.0)
        step_counts.append(float(rollout["steps"]))
        if rollout["render_path"]:
            media_paths.append(str(rollout["render_path"]))

    return {
        "eval_episode_return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval_success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
        "eval_collision_rate": float(np.mean(unsafe_flags)) if unsafe_flags else 0.0,
        "eval_steps_mean": float(np.mean(step_counts)) if step_counts else 0.0,
        "eval_media": media_paths,
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


def main() -> None:
    args = _parse_args()
    cfg: Dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42) if args.seed < 0 else args.seed)
    _seed_all(seed)

    _, obs = _fixed_env(cfg, args, seed=seed)
    agent_id = 0
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

    output_root = Path(args.output_root)
    run_dir = _make_run_dir(output_root=output_root, run_name=args.run_name or "high_level_only")
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
        success_flags: List[float] = []
        unsafe_flags: List[float] = []
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
            )
            episode_returns.append(float(rollout["episode_return"]))
            reached = bool(rollout["reached_goal"])
            unsafe = bool(rollout["unsafe"])
            success_flags.append(1.0 if (reached and not unsafe) else 0.0)
            unsafe_flags.append(1.0 if unsafe else 0.0)
            transitions.extend(rollout["transitions"])

        update_stats = high_updater.update(transitions)
        row: Dict[str, float] = {
            "iteration": float(itr),
            "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "train_success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
            "train_collision_rate": float(np.mean(unsafe_flags)) if unsafe_flags else 0.0,
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
            )
            last_eval = eval_stats
            row.update({k: float(v) for k, v in eval_stats.items() if k != "eval_media"})
            print(
                f"[iter {itr}/{total_iterations}] "
                f"train_ret={row['episode_return_mean']:.4f} "
                f"train_succ={row['train_success_rate']:.4f} "
                f"eval_succ={float(eval_stats['eval_success_rate']):.4f} "
                f"eval_coll={float(eval_stats['eval_collision_rate']):.4f} "
                f"entropy={row['high_entropy']:.4f} "
                f"ent_coef={row['high_entropy_coef']:.4f}"
            )
        else:
            print(
                f"[iter {itr}/{total_iterations}] "
                f"train_ret={row['episode_return_mean']:.4f} "
                f"train_succ={row['train_success_rate']:.4f} "
                f"entropy={row['high_entropy']:.4f} "
                f"ent_coef={row['high_entropy_coef']:.4f}"
            )
        history.append(row)

    render_dir = run_dir / "eval_media"
    final_eval = _evaluate_policy(
        cfg=cfg,
        args=args,
        policy=high_policy,
        skills=skills,
        base_seed=seed + 900_000,
        episodes=max(1, int(args.eval_episodes)),
        deterministic=True,
        invalid_skill_penalty=invalid_skill_penalty,
        render_dir=render_dir,
        render_gif=bool(args.final_render_gif) or not bool(args.final_render_png),
    )

    torch.save(
        {
            "seed": seed,
            "config": cfg,
            "high_policy": high_policy.state_dict(),
            "high_optimizer": high_opt.state_dict(),
            "mode": "high_level_only",
        },
        run_dir / "checkpoints" / "last.pt",
    )

    _write_history_csv(run_dir / "train_history.csv", history)
    summary = {
        "seed": seed,
        "total_iterations": total_iterations,
        "episodes_per_iter": episodes_per_iter,
        "invalid_skill_penalty": invalid_skill_penalty,
        "goal_visibility_reward": float(cfg.get("train", {}).get("goal_visibility_reward", 0.0)),
        "goal_visibility_improvement_reward": float(cfg.get("train", {}).get("goal_visibility_improvement_reward", 0.0)),
        "goal_heading_alignment_penalty": float(cfg.get("train", {}).get("goal_heading_alignment_penalty", 0.0)),
        "last_eval_during_train": last_eval,
        "final_eval": final_eval,
        "run_dir": str(run_dir),
        "mode": "high_level_only",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "LATEST_RUN").write_text(str(run_dir), encoding="utf-8")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
