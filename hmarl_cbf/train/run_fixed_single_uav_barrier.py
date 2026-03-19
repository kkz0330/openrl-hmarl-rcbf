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
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required to run training") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to load training config") from exc

from hmarl_cbf.buffer import HierRolloutBuffer
from hmarl_cbf.control import (
    ConstraintBuilder,
    DifferentiableQPSolver,
    LowLevelSafeController,
    SyncCoordinator,
    TorchDifferentiableQPSolver,
)
from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.eval import EpisodeTrace, TrajectoryRenderer
from hmarl_cbf.high_level import MAPPOConfig, OnPolicyMAPPO
from hmarl_cbf.policies import HighLevelPolicy, LowLevelQPPolicy
from hmarl_cbf.skills import SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.train import TrainerHooks, TrainerSyncOnPolicy


class FixedScenarioSingleUAVEnv(MultiUAV2DEnv):
    """Single-UAV fixed scene environment: start-goal line passes through one obstacle."""

    def __init__(
        self,
        *,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle_center: np.ndarray,
        obstacle_radius: float,
        initial_velocity: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs = dict(kwargs)
        kwargs["n_agents"] = 1
        kwargs["n_obstacles"] = 1
        super().__init__(**kwargs)
        v0 = np.zeros(2, dtype=np.float32) if initial_velocity is None else np.asarray(initial_velocity, dtype=np.float32).reshape(2)
        self._fixed_states: List[Dict[str, Any]] = [
            {
                "position": np.asarray(start, dtype=np.float32).reshape(2),
                "velocity": v0,
                "goal": np.asarray(goal, dtype=np.float32).reshape(2),
                "radius": float(self.agent_radius),
            }
        ]
        self._fixed_obstacles: List[Dict[str, Any]] = [
            {
                "center": np.asarray(obstacle_center, dtype=np.float32).reshape(2),
                "radius": float(obstacle_radius),
            }
        ]

    def reset(self, *, seed: int | None = None, options: Dict[str, object] | None = None):  # type: ignore[override]
        fixed_options: Dict[str, object] = {
            "states": self._fixed_states,
            "obstacles": self._fixed_obstacles,
        }
        return super().reset(seed=seed, options=fixed_options)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one UAV in a fixed line-through-obstacle scene and print high-level skill sequence each iteration."
    )
    parser.add_argument("--config", type=str, default="configs/hmarl_cbf/default_async_onpolicy_gcbfplus.yaml")
    parser.add_argument("--output-root", type=str, default="artifacts/hmarl_cbf")
    parser.add_argument("--run-name", type=str, default="train_fixed_single_uav_barrier")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--total-iterations", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--video-interval", type=int, default=20, help="Render one skill-annotated GIF every N iterations.")
    parser.add_argument("--video-fps", type=int, default=8, help="FPS for periodic GIF rendering.")
    parser.add_argument("--start-x", type=float, default=-4.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--goal-x", type=float, default=4.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--obstacle-x", type=float, default=0.0)
    parser.add_argument("--obstacle-y", type=float, default=0.0)
    parser.add_argument("--obstacle-r", type=float, default=1.0)
    parser.add_argument("--init-vx", type=float, default=0.0)
    parser.add_argument("--init-vy", type=float, default=0.0)
    parser.add_argument(
        "--init-speed-to-goal",
        type=float,
        default=-1.0,
        help="If >=0, override init-vx/init-vy and set initial speed toward goal direction.",
    )
    parser.add_argument("--low-update-mode", type=str, default="target_regression")
    return parser.parse_args()


def _make_run_dir(output_root: Path, run_name: str) -> Path:
    if run_name:
        run_dir = output_root / run_name
    else:
        run_dir = output_root / time.strftime("run_fixed_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_media").mkdir(parents=True, exist_ok=True)
    return run_dir


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _set_high_entropy_coef(trainer: TrainerSyncOnPolicy, train_cfg: Dict[str, Any], itr: int, total_iterations: int) -> float:
    if trainer.high_level_updater is None:
        return float(train_cfg.get("high_entropy_coef_end", train_cfg.get("high_entropy_coef_start", 0.0)))
    default_coef = float(trainer.high_level_updater.config.entropy_coef)
    start = float(train_cfg.get("high_entropy_coef_start", default_coef))
    end = float(train_cfg.get("high_entropy_coef_end", start))
    decay_iters = max(1, int(train_cfg.get("high_entropy_decay_iters", total_iterations)))
    if decay_iters <= 1:
        coef = end
    else:
        alpha = min(1.0, max(0.0, float(itr - 1) / float(decay_iters - 1)))
        coef = start + (end - start) * alpha
    trainer.high_level_updater.config.entropy_coef = float(coef)
    return float(coef)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _distance_point_to_segment(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    ab = seg_b - seg_a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-8:
        return float(np.linalg.norm(point - seg_a))
    t = float(np.dot(point - seg_a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    proj = seg_a + t * ab
    return float(np.linalg.norm(point - proj))


def _build_trainer(
    cfg: Dict[str, Any],
    seed: int,
    eval_episodes: int,
    deterministic_eval: bool,
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
    initial_velocity: np.ndarray,
) -> tuple[TrainerSyncOnPolicy, Any, Any, List[str], Dict[str, Any]]:
    env_cfg = dict(cfg["env"])
    env = FixedScenarioSingleUAVEnv(
        **env_cfg,
        start=start,
        goal=goal,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
        initial_velocity=initial_velocity,
    )
    obs, _ = env.reset(seed=seed)
    agent_id = sorted(obs.keys())[0]
    high_obs = obs[agent_id]["high"]
    low_obs = obs[agent_id]["low"]
    obs_dim_high = int(high_obs.self_state.shape[0] + high_obs.goal_relative.shape[0] + high_obs.neighbor_summary.shape[0])
    obs_dim_low = int(low_obs.flat.shape[0])

    skills = build_default_skill_library(max_duration=int(cfg["skills"]["default_max_duration"]))
    n_skills = len(skills)
    skill_names = list(cfg["skills"].get("names", []))
    if (not skill_names) or (len(skill_names) != n_skills):
        skill_names = [s.name for s in skills]

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

    high_opt = torch.optim.Adam(high_policy.parameters(), lr=3e-4)
    low_opt = torch.optim.Adam(low_policy.parameters(), lr=1e-3)
    mappo = OnPolicyMAPPO(
        policy=high_policy,
        optimizer=high_opt,
        config=MAPPOConfig(**cfg["high_level_mappo"]),
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
    hooks = TrainerHooks(
        rollout_steps=int(cfg["train"]["rollout_steps"]),
        eval_interval=int(cfg["train"]["eval_interval"]),
        gamma_high=float(cfg["train"]["gamma_high"]),
        lam_high=float(cfg["train"]["lam_high"]),
        gamma_low=float(cfg["train"]["gamma_low"]),
        low_ext_reward_coef=float(cfg["train"]["low_ext_reward_coef"]),
        low_update_epochs=int(cfg["train"]["low_update_epochs"]),
        low_max_samples_per_iter=int(cfg["train"]["low_max_samples_per_iter"]),
        low_target_step_scale=float(cfg["train"]["low_target_step_scale"]),
        low_update_mode=str(cfg["train"].get("low_update_mode", "target_regression")),
        low_ppo_epochs=int(cfg["train"].get("low_ppo_epochs", cfg["train"].get("low_update_epochs", 2))),
        low_ppo_clip_ratio=float(cfg["train"].get("low_ppo_clip_ratio", 0.2)),
        low_ppo_value_coef=float(cfg["train"].get("low_ppo_value_coef", 0.5)),
        low_ppo_entropy_coef=float(cfg["train"].get("low_ppo_entropy_coef", 0.0)),
        low_ppo_max_grad_norm=float(cfg["train"].get("low_ppo_max_grad_norm", 0.5)),
        low_policy_action_std=float(cfg["train"].get("low_policy_action_std", 0.2)),
        low_normalize_advantages=bool(cfg["train"].get("low_normalize_advantages", True)),
        eval_episodes=int(eval_episodes),
        eval_deterministic=bool(deterministic_eval),
        eval_render=False,
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
        diff_qp_solver=TorchDifferentiableQPSolver(
            action_dim=int(cfg["model"]["action_dim"]),
            ecos_max_iters=int(cfg["qp"].get("ecos_max_iters", 500)),
            scs_max_iters=int(cfg["qp"].get("scs_max_iters", 10_000)),
            scs_eps=float(cfg["qp"].get("scs_eps", 1e-4)),
        ),
        high_level_updater=mappo,
        low_level_optimizer=low_opt,
        hooks=hooks,
    )
    fixed_scene = {
        "start": np.asarray(start, dtype=np.float32).reshape(2).tolist(),
        "goal": np.asarray(goal, dtype=np.float32).reshape(2).tolist(),
        "initial_velocity": np.asarray(initial_velocity, dtype=np.float32).reshape(2).tolist(),
        "obstacle_center": np.asarray(obstacle_center, dtype=np.float32).reshape(2).tolist(),
        "obstacle_radius": float(obstacle_radius),
    }
    return trainer, high_opt, low_opt, skill_names, fixed_scene


def _extract_skill_sequence_for_single_agent(trainer: TrainerSyncOnPolicy, skill_names: List[str]) -> Dict[str, Any]:
    _, high_options = trainer.buffer.snapshot()
    one_agent = [tr for tr in high_options if int(tr.agent_id) == 0]
    one_agent.sort(key=lambda tr: (int(tr.t_start), int(tr.k)))
    ids: List[int] = [int(tr.skill_id) for tr in one_agent]
    names: List[str] = [skill_names[sid] if 0 <= sid < len(skill_names) else str(sid) for sid in ids]
    starts: List[int] = [int(tr.t_start) for tr in one_agent]
    ends: List[int] = [int(tr.t_end) for tr in one_agent]
    return {
        "n_rounds": len(ids),
        "skill_ids": ids,
        "skill_names": names,
        "t_start": starts,
        "t_end": ends,
    }


def _obs_batch_single(trainer: TrainerSyncOnPolicy, obs_map: Dict[int, Dict[str, Any]], agent_ids: List[int]) -> np.ndarray:
    rows = []
    for aid in agent_ids:
        h = obs_map[aid]["high"]
        rows.append(np.concatenate([h.self_state, h.goal_relative, h.neighbor_summary], axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


def _sample_high_skills(
    trainer: TrainerSyncOnPolicy,
    obs_map: Dict[int, Dict[str, Any]],
    agent_ids: List[int],
    deterministic: bool,
) -> Dict[int, int]:
    batch = _obs_batch_single(trainer, obs_map, agent_ids)
    with torch.no_grad():
        act = trainer.high_policy.act(torch.as_tensor(batch, dtype=torch.float32), deterministic=deterministic)
    z = act["z"].detach().cpu().numpy()
    return {aid: int(z[idx]) for idx, aid in enumerate(agent_ids)}


def _render_periodic_skill_video(
    trainer: TrainerSyncOnPolicy,
    skill_names: List[str],
    out_path: Path,
    *,
    deterministic: bool,
    fps: int,
) -> str:
    obs, _ = trainer.env.reset(seed=12345)
    agent_ids = sorted(obs.keys())
    trainer.skill_runtime.reset(agent_ids)
    trainer.coordinator.reset()
    renderer = TrajectoryRenderer(world_size=float(getattr(trainer.env, "world_size", 10.0)))

    sampled = _sample_high_skills(trainer, obs, agent_ids, deterministic=deterministic)
    states0 = {s.agent_id: s for s in trainer.env.get_agent_states()}
    active = trainer.activate_round_skills(skill_map=sampled, states=states0)

    current_skill = {aid: int(active[aid]) for aid in agent_ids}
    goals = np.stack([np.asarray(s.goal, dtype=np.float32).reshape(2) for s in trainer.env.get_agent_states()], axis=0)
    trace_positions: List[np.ndarray] = []
    trace_unsafe: List[np.ndarray] = []
    frame_labels: List[str] = []

    terminated = False
    truncated = False
    max_steps = int(getattr(trainer.env, "horizon", trainer.hooks.rollout_steps))

    for t in range(max_steps):
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
        switched_agents = set(sync_res.switch_agents)
        if terminated or truncated:
            switched_agents = set(agent_ids)

        current_pos = np.stack([next_states[aid].position for aid in agent_ids], axis=0).astype(np.float32)
        unsafe_row = np.asarray([bool(info.get("unsafe_flags", {}).get(aid, False)) for aid in agent_ids], dtype=bool)
        trace_positions.append(current_pos)
        trace_unsafe.append(unsafe_row)

        sid = current_skill[agent_ids[0]]
        sname = skill_names[sid] if 0 <= sid < len(skill_names) else str(sid)
        frame_labels.append(f"t={t} skill={sname}")

        obs = next_obs
        if len(switched_agents) > 0 and not (terminated or truncated):
            sampled = _sample_high_skills(trainer, obs, list(switched_agents), deterministic=deterministic)
            states_round = {s.agent_id: s for s in trainer.env.get_agent_states()}
            changed = trainer.activate_round_skills(skill_map=sampled, states=states_round)
            for aid in switched_agents:
                current_skill[aid] = int(changed[aid])
        if terminated or truncated:
            break

    if len(trace_positions) == 0:
        raise RuntimeError("no frames collected for periodic video")

    trace = EpisodeTrace(
        positions=np.stack(trace_positions, axis=0),
        goals=goals,
        obstacles=trainer.env.get_obstacles(),
        unsafe_flags=np.stack(trace_unsafe, axis=0),
        frame_labels=frame_labels,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return renderer.render_gif(trace, out_path, fps=max(1, int(fps)))


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Force single-UAV, single-obstacle training in a fixed scenario.
    cfg["env"]["n_agents"] = 1
    cfg["env"]["n_obstacles"] = 1
    cfg["train"]["total_iterations"] = int(args.total_iterations)
    cfg["train"]["eval_interval"] = int(args.eval_interval)
    cfg["train"]["low_update_mode"] = str(args.low_update_mode).strip()

    seed = int(cfg.get("seed", 42) if args.seed < 0 else args.seed)
    _seed_all(seed)

    start = np.asarray([args.start_x, args.start_y], dtype=np.float32)
    goal = np.asarray([args.goal_x, args.goal_y], dtype=np.float32)
    obstacle_center = np.asarray([args.obstacle_x, args.obstacle_y], dtype=np.float32)
    if float(args.init_speed_to_goal) >= 0.0:
        goal_vec = goal - start
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist > 1e-6:
            initial_velocity = (goal_vec / goal_dist) * float(args.init_speed_to_goal)
        else:
            initial_velocity = np.zeros(2, dtype=np.float32)
    else:
        initial_velocity = np.asarray([args.init_vx, args.init_vy], dtype=np.float32)
    obstacle_radius = float(args.obstacle_r)

    line_dist = _distance_point_to_segment(obstacle_center, start, goal)
    if line_dist > obstacle_radius:
        raise ValueError(
            "Invalid fixed scene: start-goal straight line does not pass through obstacle. "
            f"distance={line_dist:.4f}, obstacle_radius={obstacle_radius:.4f}"
        )

    output_root = Path(args.output_root)
    run_dir = _make_run_dir(output_root=output_root, run_name=args.run_name)

    trainer, high_opt, low_opt, skill_names, fixed_scene = _build_trainer(
        cfg=cfg,
        seed=seed,
        eval_episodes=max(1, int(args.eval_episodes)),
        deterministic_eval=bool(args.deterministic_eval),
        start=start,
        goal=goal,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
        initial_velocity=initial_velocity,
    )

    cfg_snapshot = dict(cfg)
    cfg_snapshot["fixed_scene"] = fixed_scene
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg_snapshot, sort_keys=False), encoding="utf-8")

    history: List[Dict[str, float]] = []
    skill_rows: List[Dict[str, Any]] = []
    last_eval: Dict[str, float] = {}
    eval_success_history: List[float] = []
    total_iterations = int(cfg["train"]["total_iterations"])
    eval_interval = max(1, int(cfg["train"]["eval_interval"]))
    video_interval = max(1, int(args.video_interval))
    video_fps = max(1, int(args.video_fps))
    iter_video_dir = run_dir / "iter_videos"

    for itr in range(1, total_iterations + 1):
        high_entropy_coef = _set_high_entropy_coef(trainer, cfg["train"], itr, total_iterations)
        rollout = trainer.collect_rollout()
        seq = _extract_skill_sequence_for_single_agent(trainer, skill_names)
        skill_rows.append(
            {
                "iteration": int(itr),
                "n_rounds": int(seq["n_rounds"]),
                "skill_ids": json.dumps(seq["skill_ids"], ensure_ascii=False),
                "skill_names": json.dumps(seq["skill_names"], ensure_ascii=False),
                "t_start": json.dumps(seq["t_start"], ensure_ascii=False),
                "t_end": json.dumps(seq["t_end"], ensure_ascii=False),
            }
        )
        print(
            f"[iter {itr}/{total_iterations}] "
            f"high_skill_seq={seq['skill_names']}"
        )

        low = trainer.update_low_level()
        high = trainer.update_high_level()

        if itr % video_interval == 0:
            vid_path = iter_video_dir / f"iter_{itr:04d}.gif"
            try:
                rendered = _render_periodic_skill_video(
                    trainer=trainer,
                    skill_names=skill_names,
                    out_path=vid_path,
                    deterministic=bool(args.deterministic_eval),
                    fps=video_fps,
                )
                print(f"[iter {itr}/{total_iterations}] periodic_video={rendered}")
            except Exception as exc:
                print(f"[iter {itr}/{total_iterations}] periodic_video_failed={exc}")
        row = {
            "iteration": float(itr),
            "steps_collected": float(rollout.get("steps_collected", 0.0)),
            "episode_return_mean": float(rollout.get("episode_return_mean", 0.0)),
            "safe_reach_ratio": float(rollout.get("safe_reach_ratio", 0.0)),
            "skill_entropy_norm": float(rollout.get("skill_entropy_norm", 0.0)),
            "top1_skill_ratio": float(rollout.get("top1_skill_ratio", 0.0)),
            "high_div_bonus_mean": float(rollout.get("high_div_bonus_mean", 0.0)),
            "conv_eval_success_delta_w5": float("nan"),
            "high_entropy_coef": float(high_entropy_coef),
            "high_samples": float(rollout.get("high_samples", 0.0)),
            "low_samples": float(rollout.get("low_samples", 0.0)),
            "loss_high_total": float(high.get("loss_total", 0.0)),
            "loss_low_mean": float(low.get("loss_mean", 0.0)),
            "loss_low_actor": float(low.get("loss_actor", 0.0)),
            "loss_low_value": float(low.get("loss_value", 0.0)),
            "low_entropy": float(low.get("entropy", 0.0)),
        }
        if itr == 1 or itr % eval_interval == 0 or itr == total_iterations:
            last_eval = trainer.evaluate()
            row.update({k: float(v) for k, v in last_eval.items()})
            eval_success_history.append(float(last_eval.get("eval_success_rate", 0.0)))
            if len(eval_success_history) >= 10:
                prev = float(np.mean(eval_success_history[-10:-5]))
                recent = float(np.mean(eval_success_history[-5:]))
                row["conv_eval_success_delta_w5"] = abs(recent - prev)
            print(
                f"[iter {itr}/{total_iterations}] "
                f"ret={row['episode_return_mean']:.4f} "
                f"safe={row['safe_reach_ratio']:.4f} "
                f"skillH={row['skill_entropy_norm']:.4f} "
                f"top1={row['top1_skill_ratio']:.4f} "
                f"succ={float(last_eval.get('eval_success_rate', 0.0)):.4f} "
                f"coll={float(last_eval.get('eval_collision_rate', 0.0)):.4f} "
                f"conv={row['conv_eval_success_delta_w5']:.4f}"
            )
        history.append(row)

    trainer.hooks.eval_episodes = 1
    trainer.hooks.eval_render = True
    trainer.hooks.eval_render_gif = True
    trainer.hooks.eval_render_dir = str(run_dir / "eval_media")
    final_eval = trainer.evaluate()

    torch.save(
        {
            "seed": seed,
            "config": cfg_snapshot,
            "high_policy": trainer.high_policy.state_dict(),
            "low_policy": trainer.low_policy.state_dict(),
            "high_optimizer": high_opt.state_dict(),
            "low_optimizer": low_opt.state_dict(),
        },
        run_dir / "checkpoints" / "last.pt",
    )
    _write_csv(run_dir / "train_history.csv", history)
    _write_csv(run_dir / "high_skill_sequence_per_iter.csv", skill_rows)

    summary = {
        "seed": seed,
        "total_iterations": total_iterations,
        "final_eval": final_eval,
        "last_eval_during_train": last_eval,
        "run_dir": str(run_dir),
        "fixed_scene": fixed_scene,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "LATEST_RUN").write_text(str(run_dir), encoding="utf-8")

    print(f"RUN_DIR={run_dir}")
    print(f"SKILL_SEQ_CSV={run_dir / 'high_skill_sequence_per_iter.csv'}")


if __name__ == "__main__":
    main()
