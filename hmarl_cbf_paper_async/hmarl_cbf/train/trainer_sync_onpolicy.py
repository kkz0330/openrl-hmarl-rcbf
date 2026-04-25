from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional backend
    torch = None  # type: ignore[assignment]

from hmarl_cbf.buffer import HierRolloutBuffer
from hmarl_cbf.control import (
    ConstraintBuilder,
    DifferentiableQPSolver,
    LowLevelSafeController,
    SyncCoordinator,
    TorchDifferentiableQPSolver,
    build_diff_constraint_constants,
)
from hmarl_cbf.baselines import DistributedCBFBaselineController
from hmarl_cbf.env.obstacles import copy_obstacle, extract_lidar_hit_point_obstacles, normalize_obstacle, obstacle_corners
from hmarl_cbf.eval import EpisodeTrace, EvalEpisodeStats, TrajectoryRenderer, evaluate_summary
from hmarl_cbf.high_level import OnPolicyMAPPO
from hmarl_cbf.skills import SkillRuntimeManager
from hmarl_cbf.types import AgentObsHigh, AgentObsLow, AgentState, LowStepTransition, QPParam


@dataclass(slots=True)
class TrainerHooks:
    rollout_steps: int = 200
    eval_interval: int = 10
    gamma_high: float = 0.99
    lam_high: float = 0.95
    gamma_low: float = 0.99
    low_ext_reward_coef: float = 0.0
    low_reward_mix_eta: float = 0.0
    low_reward_mix_divide_by_n_agents: bool = True
    low_safety_margin_coef: float = 0.0
    low_safety_margin_h_agent: float = 0.0
    low_safety_margin_h_obstacle: float = 0.0
    high_option_progress_coef: float = 0.0
    high_option_boundary_recovery_coef: float = 0.0
    high_option_boundary_threshold: float = 0.0
    high_option_trap_relief_coef: float = 0.0
    high_option_trap_enter_coef: float = 0.0
    high_option_stuck_penalty_coef: float = 0.0
    high_option_stuck_blocked_threshold: float = 0.5
    high_option_stuck_progress_threshold: float = 0.1
    high_option_stuck_speed_threshold: float = 0.2
    high_trap_blocked_lookahead: float = 4.0
    high_trap_blocked_lateral_window: float = 3.0
    high_trap_blocked_extra_margin: float = 0.1
    low_update_epochs: int = 1
    low_max_samples_per_iter: int = 256
    low_target_step_scale: float = 0.05
    low_update_mode: str = "deterministic_diff"
    low_ppo_epochs: int = 2
    low_ppo_clip_ratio: float = 0.2
    low_ppo_value_coef: float = 0.5
    low_ppo_entropy_coef: float = 0.0
    low_ppo_max_grad_norm: float = 0.5
    low_policy_action_std: float = 0.20
    low_normalize_advantages: bool = True
    low_detach_value_head_in_actor: bool = True
    low_deterministic_value_coef: float = 0.5
    low_deterministic_slack_coef: float = 0.02
    low_deterministic_cbf_slack_coef: float = 0.05
    eval_episodes: int = 3
    eval_deterministic: bool = True
    eval_render: bool = False
    eval_render_dir: str = "artifacts/eval"
    eval_render_gif: bool = False
    eval_render_fps: int = 8


class TrainerSyncOnPolicy:
    """Hierarchical on-policy trainer for shared high/low-level policies."""

    def __init__(
        self,
        env: Any,
        high_policy: Any,
        low_policy: Any,
        constraint_builder: ConstraintBuilder,
        qp_solver: DifferentiableQPSolver,
        coordinator: SyncCoordinator,
        buffer: HierRolloutBuffer,
        skill_runtime: SkillRuntimeManager | None = None,
        low_level_controller: LowLevelSafeController | None = None,
        diff_qp_solver: TorchDifferentiableQPSolver | None = None,
        high_level_updater: OnPolicyMAPPO | None = None,
        low_level_optimizer: Any | None = None,
        teacher_baseline_controller: DistributedCBFBaselineController | None = None,
        hooks: TrainerHooks | None = None,
    ) -> None:
        self.env = env
        self.high_policy = high_policy
        self.low_policy = low_policy
        self.constraint_builder = constraint_builder
        self.qp_solver = qp_solver
        self.coordinator = coordinator
        self.buffer = buffer
        self.skill_runtime = skill_runtime
        self.low_level_controller = low_level_controller
        self.diff_qp_solver = diff_qp_solver
        self.high_level_updater = high_level_updater
        self.low_level_optimizer = low_level_optimizer
        self.teacher_baseline_controller = teacher_baseline_controller
        self.hooks = hooks or TrainerHooks()
        self.training_scene_sampler: Any | None = None
        self.last_rollout_scene_name: str = "random"

    def set_skill_runtime(self, skill_runtime: SkillRuntimeManager) -> None:
        self.skill_runtime = skill_runtime

    def set_low_level_controller(self, low_level_controller: LowLevelSafeController) -> None:
        self.low_level_controller = low_level_controller

    def set_diff_qp_solver(self, diff_qp_solver: TorchDifferentiableQPSolver) -> None:
        self.diff_qp_solver = diff_qp_solver

    def set_high_level_updater(self, high_level_updater: OnPolicyMAPPO) -> None:
        self.high_level_updater = high_level_updater

    def set_low_level_optimizer(self, low_level_optimizer: Any) -> None:
        self.low_level_optimizer = low_level_optimizer

    def set_teacher_baseline_controller(self, teacher_baseline_controller: DistributedCBFBaselineController | None) -> None:
        self.teacher_baseline_controller = teacher_baseline_controller

    def set_training_scene_sampler(self, sampler: Any | None) -> None:
        self.training_scene_sampler = sampler

    @staticmethod
    def _normalized_entropy_from_counts(skill_counts: Dict[int, int]) -> float:
        total = int(sum(int(v) for v in skill_counts.values()))
        n = int(len(skill_counts))
        if total <= 0 or n <= 1:
            return 0.0
        probs = np.asarray([float(v) / float(total) for v in skill_counts.values()], dtype=np.float32)
        probs = np.clip(probs, 1e-12, 1.0)
        entropy = float(-np.sum(probs * np.log(probs)))
        return float(entropy / np.log(float(n)))

    @staticmethod
    def _boundary_clearance(state: AgentState, world_size: float) -> float:
        pos = np.asarray(state.position, dtype=np.float32).reshape(2)
        return float(world_size - max(abs(float(pos[0])), abs(float(pos[1]))))

    def _compute_blocked_score(
        self,
        state: AgentState,
        obstacles: List[Dict[str, Any]],
        d_safe_obs: float | None = None,
    ) -> float:
        if len(obstacles) == 0:
            return 0.0
        pos = np.asarray(state.position, dtype=np.float32).reshape(2)
        goal = np.asarray(state.goal, dtype=np.float32).reshape(2)
        goal_vec = goal - pos
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist <= 1e-6:
            return 0.0

        e_goal = goal_vec / goal_dist
        e_perp = np.asarray([-e_goal[1], e_goal[0]], dtype=np.float32)
        lookahead = float(max(1e-3, self.hooks.high_trap_blocked_lookahead))
        lateral_window = float(max(1e-3, self.hooks.high_trap_blocked_lateral_window))
        extra_margin = float(max(0.0, self.hooks.high_trap_blocked_extra_margin))
        safe_obs = float(self.constraint_builder.d_safe_obs if d_safe_obs is None else d_safe_obs)
        required_width = 2.0 * float(state.radius + safe_obs + extra_margin)

        intervals: List[tuple[float, float]] = []
        for obs in obstacles:
            obs_norm = normalize_obstacle(obs)
            inflated = float(state.radius + safe_obs + extra_margin)
            if obs_norm["type"] in ("circle", "point"):
                center = np.asarray(obs_norm["center"], dtype=np.float32).reshape(2)
                radius = float(obs_norm["radius"])
                rel = center - pos
                longitudinal = float(np.dot(rel, e_goal))
                if longitudinal <= 0.0 or longitudinal > lookahead:
                    continue
                lateral = float(np.dot(rel, e_perp))
                left = max(-lateral_window, lateral - (radius + inflated))
                right = min(lateral_window, lateral + (radius + inflated))
            else:
                corners = obstacle_corners(obs_norm)
                rel_corners = corners - pos.reshape(1, 2)
                longitudinal_vals = rel_corners @ e_goal.reshape(2, 1)
                lateral_vals = rel_corners @ e_perp.reshape(2, 1)
                longitudinal_min = float(np.min(longitudinal_vals)) - inflated
                longitudinal_max = float(np.max(longitudinal_vals)) + inflated
                if longitudinal_max <= 0.0 or longitudinal_min > lookahead:
                    continue
                left = max(-lateral_window, float(np.min(lateral_vals)) - inflated)
                right = min(lateral_window, float(np.max(lateral_vals)) + inflated)
            if right <= -lateral_window or left >= lateral_window:
                continue
            intervals.append((left, right))

        if len(intervals) == 0:
            return 0.0

        intervals.sort(key=lambda item: item[0])
        merged: List[List[float]] = []
        for left, right in intervals:
            if not merged or left > merged[-1][1]:
                merged.append([left, right])
            else:
                merged[-1][1] = max(merged[-1][1], right)

        max_gap = 0.0
        cursor = -lateral_window
        for left, right in merged:
            max_gap = max(max_gap, left - cursor)
            cursor = max(cursor, right)
        max_gap = max(max_gap, lateral_window - cursor)

        return float(np.clip((required_width - max_gap) / max(required_width, 1e-6), 0.0, 1.0))

    def activate_round_skills(
        self,
        skill_map: Dict[int, int],
        states: Dict[int, AgentState],
        extra_ctx: Dict[str, Any] | None = None,
    ) -> Dict[int, int]:
        if self.skill_runtime is None:
            raise RuntimeError("skill runtime manager is not set")

        actual_map: Dict[int, int] = {}
        for agent_id, preferred in skill_map.items():
            state = states[agent_id]

            try:
                self.skill_runtime.activate_skill(
                    agent_id=agent_id,
                    skill_id=int(preferred),
                    state=state,
                    extra_ctx=extra_ctx,
                )
                actual_map[agent_id] = int(preferred)
                continue
            except ValueError:
                pass

            activated = False
            for candidate in sorted(self.skill_runtime.skill_by_id.keys()):
                try:
                    self.skill_runtime.activate_skill(
                        agent_id=agent_id,
                        skill_id=int(candidate),
                        state=state,
                        extra_ctx=extra_ctx,
                    )
                    actual_map[agent_id] = int(candidate)
                    activated = True
                    break
                except ValueError:
                    continue
            if not activated:
                raise RuntimeError(f"agent {agent_id} cannot activate any skill in current state")
        return actual_map

    def start_high_option(
        self,
        k: int,
        agent_id: int,
        t_start: int,
        obs_high: AgentObsHigh,
        skill_id: int,
        logp: float,
        value: float,
        info: Dict[str, Any] | None = None,
    ) -> None:
        self.buffer.start_high_option(
            k=k,
            agent_id=agent_id,
            t_start=t_start,
            obs_high=obs_high,
            skill_id=skill_id,
            logp=logp,
            value=value,
            info=info,
        )

    def close_high_option(
        self,
        agent_id: int,
        t_end: int,
        return_ext: float,
        done: bool,
        sync_switch: bool,
        info: Dict[str, Any] | None = None,
    ) -> None:
        self.buffer.close_high_option(
            agent_id=agent_id,
            t_end=t_end,
            return_ext=return_ext,
            done=done,
            sync_switch=sync_switch,
            info=info,
        )

    def record_low_step(self, transition: LowStepTransition) -> None:
        self.buffer.add_low_step(transition)

    def evaluate_skill_step(
        self,
        states: Dict[int, AgentState],
        obs_low: Dict[int, AgentObsLow],
        actions: Dict[int, Any],
        runtime_ctx: Dict[str, Any] | None = None,
    ) -> Dict[int, bool]:
        if self.skill_runtime is None:
            raise RuntimeError("skill runtime manager is not set")
        outputs = self.skill_runtime.step_all(
            states=states,
            obs_low=obs_low,
            executed_actions=actions,
            runtime_ctx=runtime_ctx,
        )
        return {agent_id: out.beta for agent_id, out in outputs.items()}

    def compute_safe_actions(
        self,
        states: Dict[int, AgentState],
        obs_low: Dict[int, AgentObsLow],
        obstacles: List[Dict[str, Any]],
        runtime_ctx: Dict[str, Any] | None = None,
    ) -> tuple[Dict[int, Any], Dict[int, Any]]:
        if self.skill_runtime is None:
            raise RuntimeError("skill runtime manager is not set")
        if self.low_level_controller is None:
            raise RuntimeError("low level controller is not set")
        skill_targets = self.skill_runtime.control_targets(states=states, obs_low=obs_low, runtime_ctx=runtime_ctx)
        return self.low_level_controller.solve_batch(
            states=states,
            obs_low=obs_low,
            skill_targets=skill_targets,
            obstacles=obstacles,
        )

    def _is_low_update_ppo(self) -> bool:
        return str(self.hooks.low_update_mode).strip().lower() in {"onpolicy_ppo", "low_ppo", "stochastic_phi_ppo"}

    def _is_low_update_deterministic(self) -> bool:
        return str(self.hooks.low_update_mode).strip().lower() in {
            "deterministic_diff",
            "deterministic_hf",
            "target_regression",
            "reference_regression",
            "reference_pretrain",
        }

    def _is_low_update_reference_regression(self) -> bool:
        return str(self.hooks.low_update_mode).strip().lower() in {
            "reference_regression",
            "reference_pretrain",
        }

    def _extract_local_context_from_info(
        self,
        tr: LowStepTransition,
    ) -> tuple[AgentState, List[AgentState], List[Dict[str, Any]], Dict[str, Any]]:
        info = dict(tr.info or {})
        pos = np.asarray(tr.obs_low.self_state[:2], dtype=np.float32)
        vel = np.asarray(tr.obs_low.self_state[2:4], dtype=np.float32)
        goal = pos + np.asarray(tr.obs_low.goal_relative, dtype=np.float32)
        state = AgentState(agent_id=tr.agent_id, position=pos, velocity=vel, goal=goal)

        neighbors: List[AgentState] = []
        for item in list(info.get("perceived_neighbors", [])):
            if isinstance(item, AgentState):
                neighbors.append(
                    AgentState(
                        agent_id=int(item.agent_id),
                        position=np.asarray(item.position, dtype=np.float32).reshape(2),
                        velocity=np.asarray(item.velocity, dtype=np.float32).reshape(2),
                        goal=np.asarray(item.goal, dtype=np.float32).reshape(2),
                        radius=float(item.radius),
                    )
                )
            else:
                d = dict(item)
                neighbors.append(
                    AgentState(
                        agent_id=int(d.get("agent_id", -1)),
                        position=np.asarray(d["position"], dtype=np.float32).reshape(2),
                        velocity=np.asarray(d["velocity"], dtype=np.float32).reshape(2),
                        goal=np.asarray(d["goal"], dtype=np.float32).reshape(2),
                        radius=float(d.get("radius", 0.2)),
                    )
                )

        obstacles: List[Dict[str, Any]] = []
        for item in list(info.get("perceived_obstacles", [])):
            obstacles.append(copy_obstacle(dict(item)))

        safety_constraints = dict(info.get("safety_constraints", {}))
        return state, neighbors, obstacles, safety_constraints

    def _build_diff_constants(
        self,
        state_i: AgentState,
        neighbors: List[AgentState],
        obstacles: List[Dict[str, Any]],
        safety_constraints: Dict[str, Any],
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        overrides = dict(safety_constraints or {})
        use_input_bounds = bool(overrides.get("use_input_bounds", True))
        if use_input_bounds:
            u_min = np.asarray(overrides.get("u_min", self.constraint_builder.u_min), dtype=np.float32).reshape(2)
            u_max = np.asarray(overrides.get("u_max", self.constraint_builder.u_max), dtype=np.float32).reshape(2)
        else:
            unbounded = float(overrides.get("unbounded_action_limit", 1.0e6))
            u_min = np.asarray([-unbounded, -unbounded], dtype=np.float32)
            u_max = np.asarray([unbounded, unbounded], dtype=np.float32)

        constants = build_diff_constraint_constants(
            state_i=state_i,
            neighbors=neighbors,
            obstacles=obstacles,
            d_min_agent=float(overrides.get("d_min_agent", self.constraint_builder.d_min_agent)),
            d_safe_obs=float(overrides.get("d_safe_obs", self.constraint_builder.d_safe_obs)),
            cbf_mode=str(overrides.get("cbf_mode", "distributed_ecbf")),
            cbf_u_max=float(overrides.get("cbf_u_max", max(np.max(np.abs(u_min)), np.max(np.abs(u_max)), 1e-3))),
            cbf_share_agent=float(overrides.get("cbf_share_agent", 0.5)),
            cbf_share_obs=float(overrides.get("cbf_share_obs", 1.0)),
            cbf_eps=float(overrides.get("cbf_eps", 1e-4)),
            boundary_cbf=bool(overrides.get("boundary_cbf", False)),
            world_size=float(overrides.get("world_size", getattr(self.env, "world_size", 0.0))),
            boundary_margin=float(overrides.get("boundary_margin", 0.0)),
            rect_corner_margin_enabled=bool(overrides.get("rect_corner_margin_enabled", False)),
            rect_base_margin_extra=float(overrides.get("rect_base_margin_extra", 0.0)),
            rect_corner_margin_max=float(overrides.get("rect_corner_margin_max", 0.0)),
            rect_corner_proximity_distance=float(overrides.get("rect_corner_proximity_distance", 0.4)),
            rect_corner_speed_min=float(overrides.get("rect_corner_speed_min", 0.05)),
            rect_corner_alignment_power=float(overrides.get("rect_corner_alignment_power", 1.0)),
            rect_dual_edge_cbf_enabled=bool(overrides.get("rect_dual_edge_cbf_enabled", False)),
            rect_dual_edge_proximity_distance=float(overrides.get("rect_dual_edge_proximity_distance", 0.0)),
            rect_smooth_tau=float(overrides.get("rect_smooth_tau", 0.1)),
            lidar_cbf_top_k=int(overrides.get("lidar_cbf_top_k", self.constraint_builder.lidar_cbf_config.get("top_k", 0))),
            robust_cbf=bool(overrides.get("robust_cbf", False)),
            disturbance_accel_max=float(overrides.get("disturbance_accel_max", 0.0)),
            relative_disturbance_accel_max=float(overrides.get("relative_disturbance_accel_max", 0.0)),
            u_min=u_min,
            u_max=u_max,
        )
        return constants, u_min, u_max

    @staticmethod
    def _flatten_qp_param_torch(qp_param_raw: QPParam) -> QPParam:
        return QPParam(
            H_mat=qp_param_raw.H_mat.reshape(2, 2),
            f_lin=qp_param_raw.f_lin.reshape(-1),
            w_clf=qp_param_raw.w_clf.reshape(-1),
            w_cbf=qp_param_raw.w_cbf.reshape(-1),
            cbf_slack_max=qp_param_raw.cbf_slack_max.reshape(-1),
            cbf_k0=qp_param_raw.cbf_k0.reshape(-1),
            cbf_k1=qp_param_raw.cbf_k1.reshape(-1),
            clf_k=qp_param_raw.clf_k.reshape(-1),
            hocbf_gamma_h=(
                qp_param_raw.hocbf_gamma_h.reshape(-1)
                if getattr(qp_param_raw, "hocbf_gamma_h", None) is not None
                else None
            ),
            hocbf_gamma_hdot=(
                qp_param_raw.hocbf_gamma_hdot.reshape(-1)
                if getattr(qp_param_raw, "hocbf_gamma_hdot", None) is not None
                else None
            ),
        )

    @staticmethod
    def _flatten_qp_param_numpy(qp_param_raw: QPParam) -> QPParam:
        return QPParam(
            H_mat=np.asarray(qp_param_raw.H_mat.detach().cpu().numpy(), dtype=np.float32).reshape(2, 2),
            f_lin=np.asarray(qp_param_raw.f_lin.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            w_clf=np.asarray(qp_param_raw.w_clf.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            w_cbf=np.asarray(qp_param_raw.w_cbf.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            cbf_slack_max=np.asarray(qp_param_raw.cbf_slack_max.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            cbf_k0=np.asarray(qp_param_raw.cbf_k0.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            cbf_k1=np.asarray(qp_param_raw.cbf_k1.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            clf_k=np.asarray(qp_param_raw.clf_k.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            hocbf_gamma_h=(
                np.asarray(qp_param_raw.hocbf_gamma_h.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
                if getattr(qp_param_raw, "hocbf_gamma_h", None) is not None
                else None
            ),
            hocbf_gamma_hdot=(
                np.asarray(qp_param_raw.hocbf_gamma_hdot.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
                if getattr(qp_param_raw, "hocbf_gamma_hdot", None) is not None
                else None
            ),
        )

    def _compute_safe_actions_for_low_ppo(
        self,
        states: Dict[int, AgentState],
        obs_low: Dict[int, AgentObsLow],
        obstacles: List[Dict[str, Any]],
        runtime_ctx: Dict[str, Any] | None = None,
    ) -> tuple[Dict[int, Any], Dict[int, Any], Dict[int, Dict[str, Any]]]:
        if torch is None:
            raise RuntimeError("PyTorch is required for on-policy low-level updates")
        if self.skill_runtime is None:
            raise RuntimeError("skill runtime manager is not set")
        if self.low_level_controller is None:
            raise RuntimeError("low level controller is not set")
        if self.low_policy is None or not hasattr(self.low_policy, "sample_qp_params"):
            raise RuntimeError("low_policy must provide sample_qp_params(...) for stochastic phi PPO")

        skill_targets = self.skill_runtime.control_targets(states=states, obs_low=obs_low, runtime_ctx=runtime_ctx)
        agent_ids = sorted(states.keys())

        actions: Dict[int, Any] = {}
        outputs: Dict[int, Any] = {}
        per_step_stats: Dict[int, Dict[str, Any]] = {}

        for aid in agent_ids:
            state_i = states[aid]
            target = dict(skill_targets[aid])
            skill_id = int(target["skill_id"])
            safety_constraints = dict(target.get("safety_constraints", {}))
            neighbors_all = [state_j for other_id, state_j in states.items() if other_id != aid]
            neighbors = self.low_level_controller._filter_neighbors(state_i=state_i, neighbors=neighbors_all)
            obstacles_local = self.low_level_controller._filter_obstacles(
                state_i=state_i,
                obstacles=obstacles,
                obs_low=obs_low[aid],
            )

            obs_tensor = torch.as_tensor(obs_low[aid].flat, dtype=torch.float32).unsqueeze(0)
            skill_tensor = torch.as_tensor([skill_id], dtype=torch.long)
            with torch.no_grad():
                sample_out = self.low_policy.sample_qp_params(obs_tensor, skill_tensor, deterministic=False)
                qp_param_np = self._flatten_qp_param_numpy(sample_out["qp_param"])
                out = self.low_level_controller.solve_for_agent(
                    agent_id=aid,
                    state_i=state_i,
                    neighbors=neighbors,
                    obstacles=obstacles_local,
                    obs_low=obs_low[aid],
                    skill_id=skill_id,
                    safety_constraints=safety_constraints,
                    qp_param_override=qp_param_np,
                )
                actions[aid] = np.asarray(out.solution.action, dtype=np.float32).reshape(2)
                outputs[aid] = out
                value = (
                    self.low_policy.low_value(obs_tensor, skill_tensor).reshape(1)
                    if hasattr(self.low_policy, "low_value")
                    else torch.zeros((1,), dtype=torch.float32)
                )
                per_step_stats[aid] = {
                    "low_logp": float(sample_out["logp"][0].detach().cpu().item()),
                    "low_value": float(value[0].detach().cpu().item()),
                    "low_entropy": float(sample_out["entropy"][0].detach().cpu().item()),
                    "low_phi_sample": np.asarray(sample_out["phi"][0].detach().cpu().numpy(), dtype=np.float32).copy(),
                    "low_phi_mean": np.asarray(sample_out["mu"][0].detach().cpu().numpy(), dtype=np.float32).copy(),
                    "low_phi_log_std": np.asarray(sample_out["log_std"][0].detach().cpu().numpy(), dtype=np.float32).copy(),
                }
        return actions, outputs, per_step_stats

    def backward_low_level_diff_step(
        self,
        state_i: AgentState,
        neighbors: List[AgentState],
        obstacles: List[Dict[str, Any]],
        obs_low: AgentObsLow,
        skill_id: int,
        target_action: np.ndarray,
        optimizer: Any,
        d_min_agent: float = 0.6,
        d_safe_obs: float = 0.6,
        constraint_overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        if torch is None:
            raise RuntimeError("PyTorch is required for differentiable low-level updates")
        if self.diff_qp_solver is None:
            raise RuntimeError("differentiable QP solver is not set")
        if self.low_policy is None or not hasattr(self.low_policy, "forward"):
            raise RuntimeError("low_policy must be a torch module for differentiable updates")

        overrides = dict(constraint_overrides or {})
        obstacles_local = obstacles
        if bool(overrides.get("lidar_obstacle_cbf_enabled", False)):
            obstacles_local = extract_lidar_hit_point_obstacles(
                scan=obs_low.lidar_scan,
                max_range=float(overrides.get("obstacle_perception_range", obs_low.lidar_scan.max_range)),
                point_radius=float(overrides.get("lidar_cbf_point_radius", 0.0)),
                top_k=int(overrides.get("lidar_cbf_top_k", 0)) if int(overrides.get("lidar_cbf_top_k", 0)) > 0 else None,
            )
        use_input_bounds = bool(overrides.get("use_input_bounds", True))
        if use_input_bounds:
            u_min = np.asarray(overrides.get("u_min", self.constraint_builder.u_min), dtype=np.float32).reshape(2)
            u_max = np.asarray(overrides.get("u_max", self.constraint_builder.u_max), dtype=np.float32).reshape(2)
        else:
            unbounded = float(overrides.get("unbounded_action_limit", 1.0e6))
            u_min = np.asarray([-unbounded, -unbounded], dtype=np.float32)
            u_max = np.asarray([unbounded, unbounded], dtype=np.float32)

        constants = build_diff_constraint_constants(
            state_i=state_i,
            neighbors=neighbors,
            obstacles=obstacles_local,
            d_min_agent=d_min_agent,
            d_safe_obs=d_safe_obs,
            cbf_mode=str(overrides.get("cbf_mode", "distributed_ecbf")),
            cbf_u_max=float(overrides.get("cbf_u_max", max(np.max(np.abs(u_min)), np.max(np.abs(u_max)), 1e-3))),
            cbf_share_agent=float(overrides.get("cbf_share_agent", 0.5)),
            cbf_share_obs=float(overrides.get("cbf_share_obs", 1.0)),
            cbf_eps=float(overrides.get("cbf_eps", 1e-4)),
            boundary_cbf=bool(overrides.get("boundary_cbf", False)),
            world_size=float(overrides.get("world_size", getattr(self.env, "world_size", 0.0))),
            boundary_margin=float(overrides.get("boundary_margin", 0.0)),
            rect_corner_margin_enabled=bool(overrides.get("rect_corner_margin_enabled", False)),
            rect_base_margin_extra=float(overrides.get("rect_base_margin_extra", 0.0)),
            rect_corner_margin_max=float(overrides.get("rect_corner_margin_max", 0.0)),
            rect_corner_proximity_distance=float(overrides.get("rect_corner_proximity_distance", 0.4)),
            rect_corner_speed_min=float(overrides.get("rect_corner_speed_min", 0.05)),
            rect_corner_alignment_power=float(overrides.get("rect_corner_alignment_power", 1.0)),
            rect_dual_edge_cbf_enabled=bool(overrides.get("rect_dual_edge_cbf_enabled", False)),
            rect_dual_edge_proximity_distance=float(overrides.get("rect_dual_edge_proximity_distance", 0.0)),
            rect_smooth_tau=float(overrides.get("rect_smooth_tau", 0.1)),
            u_min=u_min,
            u_max=u_max,
        )
        obs_tensor = torch.as_tensor(obs_low.flat, dtype=torch.float32).unsqueeze(0)
        skill_tensor = torch.as_tensor([skill_id], dtype=torch.long)
        qp_param = self._flatten_qp_param_torch(self.low_policy(obs_tensor, skill_tensor))
        out = self.diff_qp_solver.solve(qp_param, constants)
        target = torch.as_tensor(np.asarray(target_action, dtype=np.float32).reshape(2), dtype=torch.float32)
        loss = 0.5 * torch.sum((out.action - target) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        return {
            "loss": float(loss.detach().cpu().item()),
            "action_x": float(out.action[0].detach().cpu().item()),
            "action_y": float(out.action[1].detach().cpu().item()),
            "slack": float(out.slack[0].detach().cpu().item()),
        }

    def _compute_low_values(
        self,
        obs_low: Dict[int, AgentObsLow],
        skill_ids: Dict[int, int],
    ) -> Dict[int, Dict[str, float]]:
        stats: Dict[int, Dict[str, float]] = {}
        if torch is None or self.low_policy is None or not hasattr(self.low_policy, "low_value"):
            return stats
        with torch.no_grad():
            for aid, skill_id in skill_ids.items():
                obs_tensor = torch.as_tensor(obs_low[aid].flat, dtype=torch.float32).unsqueeze(0)
                skill_tensor = torch.as_tensor([int(skill_id)], dtype=torch.long)
                value = self.low_policy.low_value(obs_tensor, skill_tensor).reshape(1)
                stats[aid] = {"low_value": float(value[0].detach().cpu().item())}
        return stats

    def finalize_rollout_buffers(
        self,
        gamma_high: float = 0.99,
        lam_high: float = 0.95,
        gamma_low: float = 0.99,
        low_ext_reward_coef: float = 0.0,
        low_reward_mix_eta: float = 0.0,
        low_reward_mix_divide_by_n_agents: bool = True,
        bootstrap_value_by_agent: Dict[int, float] | None = None,
    ) -> Dict[str, float]:
        high_stats = self.buffer.compute_high_advantages(
            gamma=gamma_high,
            lam=lam_high,
            use_gae=True,
            bootstrap_value_by_agent=bootstrap_value_by_agent,
        )
        high_adv_by_option = self.buffer.high_advantages_by_option()
        low_stats = self.buffer.compute_low_returns(
            gamma=gamma_low,
            ext_reward_coef=low_ext_reward_coef,
            reset_on_sync_switch=True,
            reward_mix_eta=low_reward_mix_eta,
            high_adv_by_option=high_adv_by_option,
            divide_high_adv_by_n_agents=low_reward_mix_divide_by_n_agents,
            n_agents=int(getattr(self.env, "n_agents", 1)),
            safety_margin_coef=float(self.hooks.low_safety_margin_coef),
            safety_margin_h_agent=float(self.hooks.low_safety_margin_h_agent),
            safety_margin_h_obstacle=float(self.hooks.low_safety_margin_h_obstacle),
        )
        return {
            "high_n": high_stats["n_samples"],
            "high_adv_mean": high_stats["adv_mean"],
            "high_adv_std": high_stats["adv_std"],
            "low_n": low_stats["n_samples"],
            "low_return_mean": low_stats["return_mean"],
            "low_adv_mean": low_stats["adv_mean"],
            "low_reward_mix_mean": low_stats["reward_mix_mean"],
            "low_high_adv_mix_mean": low_stats["high_adv_mix_mean"],
        }

    def collect_rollout(self) -> Dict[str, float]:
        if torch is None:
            raise RuntimeError("PyTorch is required for rollout collection")
        if self.skill_runtime is None:
            raise RuntimeError("skill runtime manager is not set")
        if self.low_level_controller is None:
            raise RuntimeError("low level controller is not set")
        if self.high_policy is None or not hasattr(self.high_policy, "act"):
            raise RuntimeError("high policy must provide act(...)")
        if self.env is None:
            raise RuntimeError("env is not set")

        self.buffer.clear()
        reset_options: Dict[str, Any] | None = None
        scene_name = "random"
        if callable(self.training_scene_sampler):
            sampled = self.training_scene_sampler(self.env)
            if sampled is not None:
                scene_name, reset_options = sampled
        self.last_rollout_scene_name = str(scene_name)
        obs, _ = self.env.reset(options=reset_options)
        agent_ids = sorted(obs.keys())
        self.skill_runtime.reset(agent_ids)
        self.coordinator.reset()

        ep_return_ext = {aid: 0.0 for aid in agent_ids}
        round_return_ext = {aid: 0.0 for aid in agent_ids}
        round_discount = {aid: 1.0 for aid in agent_ids}
        world_size = float(getattr(self.env, "world_size", 0.0))
        option_start_goal_dist = {
            aid: float(np.linalg.norm(state.goal - state.position))
            for aid, state in {s.agent_id: s for s in self.env.get_agent_states()}.items()
        }
        option_start_boundary_clearance = {
            aid: self._boundary_clearance(state, world_size)
            for aid, state in {s.agent_id: s for s in self.env.get_agent_states()}.items()
        }
        option_start_blocked_score = {
            aid: self._compute_blocked_score(state, self.env.get_obstacles())
            for aid, state in {s.agent_id: s for s in self.env.get_agent_states()}.items()
        }
        option_speed_sum = {aid: 0.0 for aid in agent_ids}
        option_speed_count = {aid: 0 for aid in agent_ids}
        done_by_agent = {aid: False for aid in agent_ids}
        reached_any = {aid: False for aid in agent_ids}
        unsafe_any = {aid: False for aid in agent_ids}
        skill_counts_rollout = {int(sid): 0 for sid in sorted(self.skill_runtime.skill_by_id.keys())}
        last_obs = obs
        terminated = False
        truncated = False
        frozen_agents: set[int] = set()

        def _obs_batch(obs_map: Dict[int, Dict[str, Any]]) -> np.ndarray:
            rows = []
            for aid in agent_ids:
                h = obs_map[aid]["high"]
                rows.append(np.concatenate([h.self_state, h.goal_relative, h.neighbor_summary], axis=0))
            return np.stack(rows, axis=0).astype(np.float32)

        def _sample_high(obs_map: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
            batch = _obs_batch(obs_map)
            with torch.no_grad():
                act = self.high_policy.act(torch.as_tensor(batch, dtype=torch.float32), deterministic=False)
            z = act["z"].detach().cpu().numpy()
            logp = act["logp"].detach().cpu().numpy()
            value = act["value"].detach().cpu().numpy()
            out: Dict[int, Dict[str, float]] = {}
            for idx, aid in enumerate(agent_ids):
                out[aid] = {"skill_id": int(z[idx]), "logp": float(logp[idx]), "value": float(value[idx])}
            return out

        # Start first option round.
        sampled = _sample_high(obs)
        states0 = {s.agent_id: s for s in self.env.get_agent_states()}
        actual = self.activate_round_skills(
            skill_map={aid: int(sampled[aid]["skill_id"]) for aid in agent_ids},
            states=states0,
        )
        for aid in agent_ids:
            sid = int(actual[aid])
            skill_counts_rollout[sid] = int(skill_counts_rollout.get(sid, 0)) + 1
        for aid in agent_ids:
            self.start_high_option(
                k=int(self.coordinator.option_k[aid]),
                agent_id=aid,
                t_start=self.coordinator.t,
                obs_high=obs[aid]["high"],
                skill_id=int(actual[aid]),
                logp=float(sampled[aid]["logp"]),
                value=float(sampled[aid]["value"]),
                info={
                    "round_start": True,
                    "goal_dist_start": float(option_start_goal_dist[aid]),
                    "boundary_clearance_start": float(option_start_boundary_clearance[aid]),
                    "blocked_score_start": float(option_start_blocked_score[aid]),
                },
            )

        steps_collected = 0
        for _ in range(self.hooks.rollout_steps):
            states = {s.agent_id: s for s in self.env.get_agent_states()}
            option_k_before_step = {aid: int(self.coordinator.option_k[aid]) for aid in agent_ids}
            obs_low = {aid: obs[aid]["low"] for aid in agent_ids}
            active_agent_ids = [aid for aid in agent_ids if aid not in frozen_agents]

            actions: Dict[int, Any] = {aid: np.zeros(2, dtype=np.float32) for aid in agent_ids}
            control_outputs: Dict[int, Any] = {}
            low_step_stats: Dict[int, Dict[str, Any]] = {}
            if active_agent_ids:
                active_states = {aid: states[aid] for aid in active_agent_ids}
                active_obs_low = {aid: obs_low[aid] for aid in active_agent_ids}
                if self._is_low_update_ppo():
                    active_actions, control_outputs, low_step_stats = self._compute_safe_actions_for_low_ppo(
                        states=active_states,
                        obs_low=active_obs_low,
                        obstacles=self.env.get_obstacles(),
                    )
                elif self._is_low_update_deterministic():
                    active_actions, control_outputs = self.compute_safe_actions(
                        states=active_states,
                        obs_low=active_obs_low,
                        obstacles=self.env.get_obstacles(),
                    )
                    low_step_stats = self._compute_low_values(
                        obs_low=active_obs_low,
                        skill_ids={aid: int(control_outputs[aid].skill_id) for aid in active_agent_ids},
                    )
                else:
                    raise ValueError("unsupported paper subset low-level update mode")
                actions.update(active_actions)
                teacher_actions: Dict[int, np.ndarray] = {}
                if self._is_low_update_reference_regression() and self.teacher_baseline_controller is not None:
                    teacher_actions, _ = self.teacher_baseline_controller.solve_batch(
                        states=active_states,
                        obstacles=self.env.get_obstacles(),
                        obs_low=active_obs_low,
                    )
                for aid, teacher_action in teacher_actions.items():
                    low_step_stats.setdefault(aid, {})
                    low_step_stats[aid]["teacher_action"] = np.asarray(
                        teacher_action,
                        dtype=np.float32,
                    ).reshape(2).copy()
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            next_states = {s.agent_id: s for s in self.env.get_agent_states()}
            next_obs_low = {aid: next_obs[aid]["low"] for aid in agent_ids}
            skill_out = self.skill_runtime.step_all(
                states={aid: next_states[aid] for aid in active_agent_ids},
                obs_low={aid: next_obs_low[aid] for aid in active_agent_ids},
                executed_actions={aid: actions[aid] for aid in active_agent_ids},
            ) if active_agent_ids else {}
            beta = {aid: (bool(skill_out[aid].beta) if aid in skill_out else False) for aid in agent_ids}
            step_sync = self.coordinator.step(beta)
            forced_end = bool(terminated or truncated)
            switched_agents = set(step_sync.switch_agents)
            if forced_end:
                switched_agents = set(active_agent_ids)

            for aid in agent_ids:
                unsafe = bool(info.get("unsafe_flags", {}).get(aid, False))
                reached = bool(info.get("reach_flags", {}).get(aid, False))
                unsafe_any[aid] = bool(unsafe_any[aid] or unsafe)
                reached_any[aid] = bool(reached_any[aid] or reached)
                done = bool(unsafe or reached or terminated or truncated)
                done_by_agent[aid] = done
                if aid not in active_agent_ids:
                    continue
                low_stats = dict(low_step_stats.get(aid, {}))
                low_logp = low_stats.get("low_logp", None)
                low_value = low_stats.get("low_value", None)
                self.record_low_step(
                    LowStepTransition(
                        t=step_sync.t,
                        agent_id=aid,
                        obs_low=obs_low[aid],
                        skill_id=int(skill_out[aid].skill_id),
                        option_k=int(option_k_before_step[aid]),
                        action=np.asarray(actions[aid], dtype=np.float32).reshape(2),
                        reward_int=float(skill_out[aid].intrinsic_reward),
                        reward_ext=float(rewards[aid]),
                        done=done,
                        logp=(float(low_logp) if low_logp is not None else None),
                        value=(float(low_value) if low_value is not None else None),
                        sync_switch=bool(aid in switched_agents),
                        terminated_by_skill=bool(skill_out[aid].beta),
                        info={
                            "qp_feasible": bool(control_outputs[aid].solution.feasible),
                            "qp_status": control_outputs[aid].solution.solver_status,
                            "safety_constraints": dict(control_outputs[aid].safety_constraints),
                            "min_h_agent": float(
                                info.get("safety_metrics", {}).get(aid).min_h_agent
                                if info.get("safety_metrics", {}).get(aid) is not None
                                else 0.0
                            ),
                            "min_h_obstacle": float(
                                info.get("safety_metrics", {}).get(aid).min_h_obstacle
                                if info.get("safety_metrics", {}).get(aid) is not None
                                else 0.0
                            ),
                            "perceived_neighbors": [
                                {
                                    "agent_id": int(s.agent_id),
                                    "position": np.asarray(s.position, dtype=np.float32).reshape(2).copy(),
                                    "velocity": np.asarray(s.velocity, dtype=np.float32).reshape(2).copy(),
                                    "goal": np.asarray(s.goal, dtype=np.float32).reshape(2).copy(),
                                    "radius": float(s.radius),
                                }
                                for s in control_outputs[aid].neighbors_used
                            ],
                            "perceived_obstacles": [
                                copy_obstacle(o)
                                for o in control_outputs[aid].obstacles_used
                            ],
                            "skill_u_ref": np.asarray(skill_out[aid].u_ref_skill, dtype=np.float32).reshape(2).copy(),
                            "teacher_action": (
                                np.asarray(low_stats["teacher_action"], dtype=np.float32).reshape(2).copy()
                                if "teacher_action" in low_stats
                                else None
                            ),
                            "low_qp_H": np.asarray(control_outputs[aid].qp_param.H_mat, dtype=np.float32).reshape(2, 2).copy(),
                            "low_qp_F": np.asarray(control_outputs[aid].qp_param.f_lin, dtype=np.float32).reshape(2).copy(),
                            "low_phi_sample": (
                                np.asarray(low_step_stats[aid]["low_phi_sample"], dtype=np.float32).copy()
                                if aid in low_step_stats and "low_phi_sample" in low_step_stats[aid]
                                else None
                            ),
                            "low_phi_mean": (
                                np.asarray(low_step_stats[aid]["low_phi_mean"], dtype=np.float32).copy()
                                if aid in low_step_stats and "low_phi_mean" in low_step_stats[aid]
                                else None
                            ),
                            "low_phi_log_std": (
                                np.asarray(low_step_stats[aid]["low_phi_log_std"], dtype=np.float32).copy()
                                if aid in low_step_stats and "low_phi_log_std" in low_step_stats[aid]
                                else None
                            ),
                            "low_entropy": float(low_step_stats.get(aid, {}).get("low_entropy", 0.0)),
                            "low_cbf_slack": np.asarray(
                                control_outputs[aid].solution.cbf_slack if control_outputs[aid].solution.cbf_slack is not None else [],
                                dtype=np.float32,
                            ).reshape(-1).copy(),
                        },
                    )
                )
                round_return_ext[aid] += round_discount[aid] * float(rewards[aid])
                round_discount[aid] *= self.hooks.gamma_high
                ep_return_ext[aid] += float(rewards[aid])
                option_speed_sum[aid] += float(np.linalg.norm(next_states[aid].velocity))
                option_speed_count[aid] += 1

            newly_frozen = {
                aid for aid in active_agent_ids
                if bool(info.get("reach_flags", {}).get(aid, False))
            }
            if newly_frozen and hasattr(self.env, "freeze_agents"):
                self.env.freeze_agents(sorted(newly_frozen))
            frozen_agents.update(newly_frozen)

            close_agents = set(switched_agents) | set(newly_frozen)
            if len(close_agents) > 0:
                t_end = self.coordinator.t
                for aid in close_agents:
                    if self.buffer.has_open_high_option(aid):
                        goal_dist_end = float(np.linalg.norm(next_states[aid].goal - next_states[aid].position))
                        goal_dist_start = float(option_start_goal_dist.get(aid, goal_dist_end))
                        option_progress_bonus = float(self.hooks.high_option_progress_coef) * float(
                            goal_dist_start - goal_dist_end
                        )
                        boundary_clearance_end = self._boundary_clearance(next_states[aid], world_size)
                        boundary_clearance_start = float(
                            option_start_boundary_clearance.get(aid, boundary_clearance_end)
                        )
                        blocked_score_end = self._compute_blocked_score(next_states[aid], self.env.get_obstacles())
                        blocked_score_start = float(option_start_blocked_score.get(aid, blocked_score_end))
                        boundary_recovery_bonus = 0.0
                        threshold = float(self.hooks.high_option_boundary_threshold)
                        if threshold > 0.0 and boundary_clearance_start <= threshold:
                            boundary_recovery_bonus = float(self.hooks.high_option_boundary_recovery_coef) * float(
                                boundary_clearance_end - boundary_clearance_start
                            )
                        trap_relief_bonus = float(self.hooks.high_option_trap_relief_coef) * float(
                            blocked_score_start - blocked_score_end
                        )
                        trap_enter_penalty = float(self.hooks.high_option_trap_enter_coef) * float(
                            blocked_score_start * max(0.0, goal_dist_start - goal_dist_end)
                        )
                        option_progress = float(goal_dist_start - goal_dist_end)
                        avg_speed = float(option_speed_sum.get(aid, 0.0) / max(1, option_speed_count.get(aid, 0)))
                        stuck_penalty = 0.0
                        if (
                            float(self.hooks.high_option_stuck_penalty_coef) > 0.0
                            and blocked_score_end >= float(self.hooks.high_option_stuck_blocked_threshold)
                            and option_progress <= float(self.hooks.high_option_stuck_progress_threshold)
                            and avg_speed <= float(self.hooks.high_option_stuck_speed_threshold)
                        ):
                            stuck_penalty = float(self.hooks.high_option_stuck_penalty_coef)
                        self.close_high_option(
                            agent_id=aid,
                            t_end=t_end,
                            return_ext=(
                                round_return_ext[aid]
                                + option_progress_bonus
                                + boundary_recovery_bonus
                                + trap_relief_bonus
                                - trap_enter_penalty
                                - stuck_penalty
                            ),
                            done=done_by_agent[aid],
                            sync_switch=bool(aid in switched_agents),
                            info={
                                "forced_sync": forced_end,
                                "frozen_reached": bool(aid in newly_frozen),
                                "goal_dist_start": goal_dist_start,
                                "goal_dist_end": goal_dist_end,
                                "option_progress_bonus": option_progress_bonus,
                                "boundary_clearance_start": boundary_clearance_start,
                                "boundary_clearance_end": boundary_clearance_end,
                                "boundary_recovery_bonus": boundary_recovery_bonus,
                                "blocked_score_start": blocked_score_start,
                                "blocked_score_end": blocked_score_end,
                                "trap_relief_bonus": trap_relief_bonus,
                                "trap_enter_penalty": trap_enter_penalty,
                                "option_avg_speed": avg_speed,
                                "stuck_penalty": stuck_penalty,
                            },
                        )
                        round_return_ext[aid] = 0.0
                        round_discount[aid] = 1.0
                        option_speed_sum[aid] = 0.0
                        option_speed_count[aid] = 0

            steps_collected += 1
            last_obs = next_obs
            obs = next_obs

            if forced_end:
                break

            resample_agents = [aid for aid in switched_agents if aid not in frozen_agents]
            if len(resample_agents) > 0:
                sampled = _sample_high(obs)
                states_round = {s.agent_id: s for s in self.env.get_agent_states()}
                new_skills = {aid: int(sampled[aid]["skill_id"]) for aid in resample_agents}
                actual = self.activate_round_skills(
                    skill_map=new_skills,
                    states=states_round,
                )
                for aid in resample_agents:
                    sid = int(actual[aid])
                    skill_counts_rollout[sid] = int(skill_counts_rollout.get(sid, 0)) + 1
                    self.start_high_option(
                        k=int(step_sync.option_k[aid]),
                        agent_id=aid,
                        t_start=self.coordinator.t,
                        obs_high=obs[aid]["high"],
                        skill_id=int(actual[aid]),
                        logp=float(sampled[aid]["logp"]),
                        value=float(sampled[aid]["value"]),
                        info={
                            "goal_dist_start": float(np.linalg.norm(states_round[aid].goal - states_round[aid].position)),
                            "boundary_clearance_start": float(self._boundary_clearance(states_round[aid], world_size)),
                            "blocked_score_start": float(self._compute_blocked_score(states_round[aid], self.env.get_obstacles())),
                        },
                    )
                    option_start_goal_dist[aid] = float(np.linalg.norm(states_round[aid].goal - states_round[aid].position))
                    option_start_boundary_clearance[aid] = float(self._boundary_clearance(states_round[aid], world_size))
                    option_start_blocked_score[aid] = float(self._compute_blocked_score(states_round[aid], self.env.get_obstacles()))
                    option_speed_sum[aid] = 0.0
                    option_speed_count[aid] = 0

        for aid in agent_ids:
            if self.buffer.has_open_high_option(aid):
                final_states = {s.agent_id: s for s in self.env.get_agent_states()}
                goal_dist_end = float(np.linalg.norm(final_states[aid].goal - final_states[aid].position))
                goal_dist_start = float(option_start_goal_dist.get(aid, goal_dist_end))
                option_progress_bonus = float(self.hooks.high_option_progress_coef) * float(
                    goal_dist_start - goal_dist_end
                )
                boundary_clearance_end = self._boundary_clearance(final_states[aid], world_size)
                boundary_clearance_start = float(option_start_boundary_clearance.get(aid, boundary_clearance_end))
                blocked_score_end = self._compute_blocked_score(final_states[aid], self.env.get_obstacles())
                blocked_score_start = float(option_start_blocked_score.get(aid, blocked_score_end))
                boundary_recovery_bonus = 0.0
                threshold = float(self.hooks.high_option_boundary_threshold)
                if threshold > 0.0 and boundary_clearance_start <= threshold:
                    boundary_recovery_bonus = float(self.hooks.high_option_boundary_recovery_coef) * float(
                        boundary_clearance_end - boundary_clearance_start
                    )
                trap_relief_bonus = float(self.hooks.high_option_trap_relief_coef) * float(
                    blocked_score_start - blocked_score_end
                )
                trap_enter_penalty = float(self.hooks.high_option_trap_enter_coef) * float(
                    blocked_score_start * max(0.0, goal_dist_start - goal_dist_end)
                )
                option_progress = float(goal_dist_start - goal_dist_end)
                avg_speed = float(option_speed_sum.get(aid, 0.0) / max(1, option_speed_count.get(aid, 0)))
                stuck_penalty = 0.0
                if (
                    float(self.hooks.high_option_stuck_penalty_coef) > 0.0
                    and blocked_score_end >= float(self.hooks.high_option_stuck_blocked_threshold)
                    and option_progress <= float(self.hooks.high_option_stuck_progress_threshold)
                    and avg_speed <= float(self.hooks.high_option_stuck_speed_threshold)
                ):
                    stuck_penalty = float(self.hooks.high_option_stuck_penalty_coef)
                self.close_high_option(
                    agent_id=aid,
                    t_end=self.coordinator.t,
                    return_ext=(
                        round_return_ext[aid]
                        + option_progress_bonus
                        + boundary_recovery_bonus
                        + trap_relief_bonus
                        - trap_enter_penalty
                        - stuck_penalty
                    ),
                    done=done_by_agent[aid],
                    sync_switch=True,
                    info={
                        "cutoff_close": True,
                        "goal_dist_start": goal_dist_start,
                        "goal_dist_end": goal_dist_end,
                        "option_progress_bonus": option_progress_bonus,
                        "boundary_clearance_start": boundary_clearance_start,
                        "boundary_clearance_end": boundary_clearance_end,
                        "boundary_recovery_bonus": boundary_recovery_bonus,
                        "blocked_score_start": blocked_score_start,
                        "blocked_score_end": blocked_score_end,
                        "trap_relief_bonus": trap_relief_bonus,
                        "trap_enter_penalty": trap_enter_penalty,
                        "option_avg_speed": avg_speed,
                        "stuck_penalty": stuck_penalty,
                    },
                )

        bootstrap: Dict[int, float] = {}
        if not (terminated or truncated):
            batch = _obs_batch(last_obs)
            with torch.no_grad():
                _, value = self.high_policy.forward(torch.as_tensor(batch, dtype=torch.float32))
            value_np = value.detach().cpu().numpy()
            for idx, aid in enumerate(agent_ids):
                bootstrap[aid] = float(value_np[idx])

        post = self.finalize_rollout_buffers(
            gamma_high=self.hooks.gamma_high,
            lam_high=self.hooks.lam_high,
            gamma_low=self.hooks.gamma_low,
            low_ext_reward_coef=self.hooks.low_ext_reward_coef,
            low_reward_mix_eta=self.hooks.low_reward_mix_eta,
            low_reward_mix_divide_by_n_agents=self.hooks.low_reward_mix_divide_by_n_agents,
            bootstrap_value_by_agent=bootstrap,
        )
        n_agents = max(1, len(agent_ids))
        safe_reach_ratio = float(
            sum(1.0 for aid in agent_ids if reached_any[aid] and not unsafe_any[aid]) / n_agents
        )
        total_skill_activations = int(sum(skill_counts_rollout.values()))
        skill_entropy_norm = float(self._normalized_entropy_from_counts(skill_counts_rollout))
        top1_skill_ratio = float(
            max((int(v) for v in skill_counts_rollout.values()), default=0) / max(1, total_skill_activations)
        )
        return {
            "steps_collected": float(steps_collected),
            "episode_return_mean": float(sum(ep_return_ext.values()) / max(1, len(agent_ids))),
            "safe_reach_ratio": safe_reach_ratio,
            "high_samples": float(self.buffer.size_high()),
            "low_samples": float(self.buffer.size_low()),
            "skill_entropy_norm": skill_entropy_norm,
            "top1_skill_ratio": top1_skill_ratio,
            "high_div_bonus_mean": 0.0,
            **post,
        }

    def update_low_level(self) -> Dict[str, float]:
        if self._is_low_update_ppo():
            return self._update_low_level_onpolicy_ppo()
        if not self._is_low_update_deterministic():
            raise ValueError(
                "The paper-aligned subset only supports stochastic phi PPO or deterministic H/F updates."
            )
        return self._update_low_level_deterministic_diff()

    def _update_low_level_deterministic_diff(self) -> Dict[str, float]:
        if self.diff_qp_solver is None or self.low_level_optimizer is None:
            return {
                "n_updates": 0.0,
                "loss_mean": 0.0,
                "loss_last": 0.0,
                "low_f_mean_x": 0.0,
                "low_f_mean_y": 0.0,
                "low_h_eig_min": 0.0,
                "low_h_eig_max": 0.0,
            }
        if self.low_policy is None or not hasattr(self.low_policy, "forward"):
            return {
                "n_updates": 0.0,
                "loss_mean": 0.0,
                "loss_last": 0.0,
                "low_f_mean_x": 0.0,
                "low_f_mean_y": 0.0,
                "low_h_eig_min": 0.0,
                "low_h_eig_max": 0.0,
            }

        low_steps = self.buffer.snapshot()[0]
        if len(low_steps) == 0:
            return {"n_updates": 0.0, "loss_mean": 0.0, "loss_last": 0.0}

        selected = list(low_steps)[-self.hooks.low_max_samples_per_iter :]
        total_losses: List[float] = []
        actor_losses: List[float] = []
        value_losses: List[float] = []
        slack_losses: List[float] = []
        f_x_vals: List[float] = []
        f_y_vals: List[float] = []
        h_eig_min_vals: List[float] = []
        h_eig_max_vals: List[float] = []
        n_updates = 0
        value_coef = float(self.hooks.low_deterministic_value_coef)
        slack_coef = float(self.hooks.low_deterministic_slack_coef)
        cbf_slack_coef = float(self.hooks.low_deterministic_cbf_slack_coef)
        for _ in range(max(1, self.hooks.low_update_epochs)):
            for tr in selected:
                state, neighbors, obstacles, safety_constraints = self._extract_local_context_from_info(tr)

                if self._is_low_update_reference_regression():
                    target_action = np.asarray(
                        tr.info.get("teacher_action", tr.info.get("skill_u_ref", tr.action)),
                        dtype=np.float32,
                    ).reshape(2)
                else:
                    adv = float(
                        tr.advantage if tr.advantage is not None else tr.return_target if tr.return_target is not None else 0.0
                    )
                    goal_dir = np.asarray(tr.obs_low.goal_relative, dtype=np.float32)
                    norm = float(np.linalg.norm(goal_dir))
                    if norm > 1e-6:
                        goal_dir = goal_dir / norm
                    delta = self.hooks.low_target_step_scale * adv * goal_dir
                    target_action = np.asarray(tr.action, dtype=np.float32) + delta
                if bool(safety_constraints.get("use_input_bounds", True)):
                    target_u_min = np.asarray(
                        safety_constraints.get("u_min", self.constraint_builder.u_min), dtype=np.float32
                    ).reshape(2)
                    target_u_max = np.asarray(
                        safety_constraints.get("u_max", self.constraint_builder.u_max), dtype=np.float32
                    ).reshape(2)
                    target_action = np.clip(target_action, target_u_min, target_u_max)
                constants, _, _ = self._build_diff_constants(
                    state_i=state,
                    neighbors=neighbors,
                    obstacles=obstacles,
                    safety_constraints=safety_constraints,
                )
                obs_tensor = torch.as_tensor(tr.obs_low.flat, dtype=torch.float32).unsqueeze(0)
                skill_tensor = torch.as_tensor([int(tr.skill_id)], dtype=torch.long)
                qp_param = self._flatten_qp_param_torch(self.low_policy(obs_tensor, skill_tensor))
                f_vec = qp_param.f_lin.reshape(-1)
                h_eigs = torch.linalg.eigvalsh(qp_param.H_mat.reshape(2, 2))
                out = self.diff_qp_solver.solve(qp_param, constants)
                action_pred = out.action.reshape(2)
                target_t = torch.as_tensor(target_action, dtype=torch.float32, device=action_pred.device)
                actor_loss = 0.5 * torch.sum((action_pred - target_t) ** 2)
                slack_loss = slack_coef * torch.sum(out.slack ** 2)
                if out.cbf_slack.numel() > 0:
                    slack_loss = slack_loss + cbf_slack_coef * torch.mean(out.cbf_slack ** 2)

                value_loss = torch.zeros((), dtype=torch.float32, device=action_pred.device)
                if hasattr(self.low_policy, "low_value") and tr.return_target is not None:
                    value_pred = self.low_policy.low_value(obs_tensor, skill_tensor).reshape(1)[0].to(device=action_pred.device)
                    ret_t = torch.as_tensor(float(tr.return_target), dtype=torch.float32, device=action_pred.device)
                    value_loss = 0.5 * (value_pred - ret_t) ** 2

                total_loss = actor_loss + value_coef * value_loss + slack_loss
                self.low_level_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                self.low_level_optimizer.step()

                total_losses.append(float(total_loss.detach().cpu().item()))
                actor_losses.append(float(actor_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                slack_losses.append(float(slack_loss.detach().cpu().item()))
                f_x_vals.append(float(f_vec[0].detach().cpu().item()))
                f_y_vals.append(float(f_vec[1].detach().cpu().item()))
                h_eig_min_vals.append(float(h_eigs[0].detach().cpu().item()))
                h_eig_max_vals.append(float(h_eigs[-1].detach().cpu().item()))
                n_updates += 1

        return {
            "n_updates": float(n_updates),
            "loss_mean": float(sum(total_losses) / max(1, len(total_losses))),
            "loss_last": float(total_losses[-1] if total_losses else 0.0),
            "loss_actor": float(sum(actor_losses) / max(1, len(actor_losses))),
            "loss_value": float(sum(value_losses) / max(1, len(value_losses))),
            "loss_slack": float(sum(slack_losses) / max(1, len(slack_losses))),
            "low_f_mean_x": float(sum(f_x_vals) / max(1, len(f_x_vals))),
            "low_f_mean_y": float(sum(f_y_vals) / max(1, len(f_y_vals))),
            "low_h_eig_min": float(sum(h_eig_min_vals) / max(1, len(h_eig_min_vals))),
            "low_h_eig_max": float(sum(h_eig_max_vals) / max(1, len(h_eig_max_vals))),
        }

    def _update_low_level_onpolicy_ppo(self) -> Dict[str, float]:
        if torch is None:
            raise RuntimeError("PyTorch is required for on-policy low-level updates")
        if self.diff_qp_solver is None or self.low_level_optimizer is None:
            return {
                "n_updates": 0.0,
                "loss_mean": 0.0,
                "loss_last": 0.0,
                "low_f_mean_x": 0.0,
                "low_f_mean_y": 0.0,
                "low_h_eig_min": 0.0,
                "low_h_eig_max": 0.0,
            }
        if self.low_policy is None or not hasattr(self.low_policy, "evaluate_phi") or not hasattr(self.low_policy, "low_value"):
            raise RuntimeError("low_policy must provide evaluate_phi(...) and low_value(...) for stochastic phi PPO")

        low_steps = self.buffer.snapshot()[0]
        if len(low_steps) == 0:
            return {
                "n_updates": 0.0,
                "loss_mean": 0.0,
                "loss_last": 0.0,
                "low_f_mean_x": 0.0,
                "low_f_mean_y": 0.0,
                "low_h_eig_min": 0.0,
                "low_h_eig_max": 0.0,
            }

        selected = list(low_steps)[-self.hooks.low_max_samples_per_iter :]
        ppo_samples = [
            tr
            for tr in selected
            if tr.logp is not None
            and tr.return_target is not None
            and tr.info.get("low_phi_sample") is not None
        ]
        if len(ppo_samples) == 0:
            return {"n_updates": 0.0, "loss_mean": 0.0, "loss_last": 0.0}

        adv_np = np.asarray(
            [
                float(tr.advantage if tr.advantage is not None else tr.return_target if tr.return_target is not None else 0.0)
                for tr in ppo_samples
            ],
            dtype=np.float32,
        )
        if bool(self.hooks.low_normalize_advantages) and adv_np.size > 1:
            adv_mean = float(np.mean(adv_np))
            adv_std = float(np.std(adv_np) + 1e-6)
            adv_np = (adv_np - adv_mean) / adv_std

        clip_ratio = float(self.hooks.low_ppo_clip_ratio)
        value_coef = float(self.hooks.low_ppo_value_coef)
        entropy_coef = float(self.hooks.low_ppo_entropy_coef)
        max_grad_norm = float(self.hooks.low_ppo_max_grad_norm)
        slack_coef = float(self.hooks.low_deterministic_slack_coef)
        cbf_slack_coef = float(self.hooks.low_deterministic_cbf_slack_coef)

        loss_all: List[float] = []
        loss_actor_all: List[float] = []
        loss_value_all: List[float] = []
        entropy_all: List[float] = []
        slack_all: List[float] = []
        f_x_vals: List[float] = []
        f_y_vals: List[float] = []
        h_eig_min_vals: List[float] = []
        h_eig_max_vals: List[float] = []
        n_updates = 0

        n_epochs = max(1, int(self.hooks.low_ppo_epochs))
        for _ in range(n_epochs):
            order = np.random.permutation(len(ppo_samples))
            for idx in order:
                tr = ppo_samples[int(idx)]
                adv = float(adv_np[int(idx)])
                state, neighbors, obstacles, safety_constraints = self._extract_local_context_from_info(tr)
                constants, _, _ = self._build_diff_constants(
                    state_i=state,
                    neighbors=neighbors,
                    obstacles=obstacles,
                    safety_constraints=safety_constraints,
                )
                obs_tensor = torch.as_tensor(tr.obs_low.flat, dtype=torch.float32).unsqueeze(0)
                skill_tensor = torch.as_tensor([int(tr.skill_id)], dtype=torch.long)
                phi_sample = torch.as_tensor(
                    np.asarray(tr.info["low_phi_sample"], dtype=np.float32).reshape(1, -1),
                    dtype=torch.float32,
                )
                eval_out = self.low_policy.evaluate_phi(obs_tensor, skill_tensor, phi_sample)
                new_logp = eval_out["logp"].reshape(1)[0]
                entropy_bonus = eval_out["entropy"].reshape(1)[0]
                qp_param = self._flatten_qp_param_torch(eval_out["qp_param"])
                f_vec = qp_param.f_lin.reshape(-1)
                h_eigs = torch.linalg.eigvalsh(qp_param.H_mat.reshape(2, 2))
                qp_out = self.diff_qp_solver.solve(qp_param, constants)

                old_logp = torch.as_tensor(float(tr.logp), dtype=torch.float32, device=new_logp.device)
                adv_t = torch.as_tensor(adv, dtype=torch.float32, device=new_logp.device)
                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_t
                actor_loss = -torch.minimum(surr1, surr2)

                value_pred = self.low_policy.low_value(obs_tensor, skill_tensor).reshape(1).to(device=new_logp.device)[0]
                ret_t = torch.as_tensor(float(tr.return_target), dtype=torch.float32, device=new_logp.device)
                value_loss = 0.5 * (value_pred - ret_t) ** 2

                slack_loss = slack_coef * torch.sum(qp_out.slack ** 2)
                if qp_out.cbf_slack.numel() > 0:
                    slack_loss = slack_loss + cbf_slack_coef * torch.mean(qp_out.cbf_slack ** 2)

                total_loss = actor_loss + value_coef * value_loss - entropy_coef * entropy_bonus + slack_loss
                self.low_level_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.low_policy.parameters(), max_grad_norm)
                self.low_level_optimizer.step()

                loss_all.append(float(total_loss.detach().cpu().item()))
                loss_actor_all.append(float(actor_loss.detach().cpu().item()))
                loss_value_all.append(float(value_loss.detach().cpu().item()))
                entropy_all.append(float(entropy_bonus.detach().cpu().item()))
                slack_all.append(float(slack_loss.detach().cpu().item()))
                f_x_vals.append(float(f_vec[0].detach().cpu().item()))
                f_y_vals.append(float(f_vec[1].detach().cpu().item()))
                h_eig_min_vals.append(float(h_eigs[0].detach().cpu().item()))
                h_eig_max_vals.append(float(h_eigs[-1].detach().cpu().item()))
                n_updates += 1

        return {
            "n_updates": float(n_updates),
            "loss_mean": float(sum(loss_all) / max(1, len(loss_all))),
            "loss_last": float(loss_all[-1] if loss_all else 0.0),
            "loss_actor": float(sum(loss_actor_all) / max(1, len(loss_actor_all))),
            "loss_value": float(sum(loss_value_all) / max(1, len(loss_value_all))),
            "entropy": float(sum(entropy_all) / max(1, len(entropy_all))),
            "loss_slack": float(sum(slack_all) / max(1, len(slack_all))),
            "low_f_mean_x": float(sum(f_x_vals) / max(1, len(f_x_vals))),
            "low_f_mean_y": float(sum(f_y_vals) / max(1, len(f_y_vals))),
            "low_h_eig_min": float(sum(h_eig_min_vals) / max(1, len(h_eig_min_vals))),
            "low_h_eig_max": float(sum(h_eig_max_vals) / max(1, len(h_eig_max_vals))),
        }

    def update_high_level(self) -> Dict[str, float]:
        if self.high_level_updater is None:
            raise RuntimeError("high-level MAPPO updater is not set")
        transitions = self.buffer.snapshot()[1]
        metrics = self.high_level_updater.update(transitions)
        self.buffer.clear_high()
        return metrics

    def evaluate(self) -> Dict[str, float]:
        if torch is None:
            raise RuntimeError("PyTorch is required for evaluation")
        if self.skill_runtime is None:
            raise RuntimeError("skill runtime manager is not set")
        if self.low_level_controller is None:
            raise RuntimeError("low level controller is not set")
        if self.high_policy is None or not hasattr(self.high_policy, "act"):
            raise RuntimeError("high policy must provide act(...)")

        n_eval = max(1, int(self.hooks.eval_episodes))
        episode_stats: List[EvalEpisodeStats] = []
        renderer = TrajectoryRenderer(world_size=float(getattr(self.env, "world_size", 10.0)))

        for ep in range(n_eval):
            obs, _ = self.env.reset(seed=10_000 + ep)
            agent_ids = sorted(obs.keys())
            self.skill_runtime.reset(agent_ids)
            self.coordinator.reset()
            if hasattr(self.env, "unfreeze_all_agents"):
                self.env.unfreeze_all_agents()

            reached_any = {aid: False for aid in agent_ids}
            unsafe_any = {aid: False for aid in agent_ids}
            frozen_agents: set[int] = set()
            traj_len = {aid: 0.0 for aid in agent_ids}
            ep_return = {aid: 0.0 for aid in agent_ids}
            skill_switches = 0
            qp_total = 0
            qp_feasible = 0
            min_h_agent = float("inf")
            min_h_obs = float("inf")

            last_pos = {
                s.agent_id: np.asarray(s.position, dtype=np.float32).copy()
                for s in self.env.get_agent_states()
            }
            goals = np.stack(
                [np.asarray(s.goal, dtype=np.float32).reshape(2) for s in self.env.get_agent_states()],
                axis=0,
            )
            trace_positions: List[np.ndarray] = []
            trace_unsafe: List[np.ndarray] = []
            frame_labels: List[str] = []
            obstacles = self.env.get_obstacles()

            def _obs_batch(obs_map: Dict[int, Dict[str, Any]]) -> np.ndarray:
                rows = []
                for aid in agent_ids:
                    h = obs_map[aid]["high"]
                    rows.append(np.concatenate([h.self_state, h.goal_relative, h.neighbor_summary], axis=0))
                return np.stack(rows, axis=0).astype(np.float32)

            def _sample_high(obs_map: Dict[int, Dict[str, Any]]) -> Dict[int, int]:
                batch = _obs_batch(obs_map)
                with torch.no_grad():
                    act = self.high_policy.act(
                        torch.as_tensor(batch, dtype=torch.float32),
                        deterministic=bool(self.hooks.eval_deterministic),
                    )
                z = act["z"].detach().cpu().numpy()
                return {aid: int(z[idx]) for idx, aid in enumerate(agent_ids)}

            sampled = _sample_high(obs)
            states0 = {s.agent_id: s for s in self.env.get_agent_states()}
            self.activate_round_skills(skill_map=sampled, states=states0)

            terminated = False
            truncated = False
            max_steps = int(getattr(self.env, "horizon", self.hooks.rollout_steps))
            for _ in range(max_steps):
                states = {s.agent_id: s for s in self.env.get_agent_states()}
                obs_low = {aid: obs[aid]["low"] for aid in agent_ids}
                active_agent_ids = [aid for aid in agent_ids if aid not in frozen_agents]
                actions: Dict[int, Any] = {aid: np.zeros(2, dtype=np.float32) for aid in agent_ids}
                control_outputs: Dict[int, Any] = {}
                if active_agent_ids:
                    active_states = {aid: states[aid] for aid in active_agent_ids}
                    active_obs_low = {aid: obs_low[aid] for aid in active_agent_ids}
                    active_actions, control_outputs = self.compute_safe_actions(
                        states=active_states,
                        obs_low=active_obs_low,
                        obstacles=self.env.get_obstacles(),
                    )
                    actions.update(active_actions)
                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                next_states = {s.agent_id: s for s in self.env.get_agent_states()}
                next_obs_low = {aid: next_obs[aid]["low"] for aid in agent_ids}
                skill_out = self.skill_runtime.step_all(
                    states={aid: next_states[aid] for aid in active_agent_ids},
                    obs_low={aid: next_obs_low[aid] for aid in active_agent_ids},
                    executed_actions={aid: actions[aid] for aid in active_agent_ids},
                ) if active_agent_ids else {}
                beta = {aid: (bool(skill_out[aid].beta) if aid in skill_out else False) for aid in agent_ids}
                sync_res = self.coordinator.step(beta)
                forced_end = bool(terminated or truncated)
                switched_agents = set(sync_res.switch_agents)
                if forced_end:
                    switched_agents = set(active_agent_ids)
                if len(switched_agents) > 0:
                    skill_switches += int(len(switched_agents))

                current_pos = np.stack([next_states[aid].position for aid in agent_ids], axis=0).astype(np.float32)
                trace_positions.append(current_pos)
                trace_unsafe.append(np.asarray([bool(info.get("unsafe_flags", {}).get(aid, False)) for aid in agent_ids], dtype=bool))
                wind_accel = np.asarray(info.get("wind_accel", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
                frame_labels.append(
                    f"t={len(trace_positions)-1}  wind=({wind_accel[0]:+0.2f}, {wind_accel[1]:+0.2f})  |w|={float(np.linalg.norm(wind_accel)):.2f}"
                )

                for aid in agent_ids:
                    ep_return[aid] += float(rewards[aid])
                    reached = bool(info.get("reach_flags", {}).get(aid, False))
                    unsafe = bool(info.get("unsafe_flags", {}).get(aid, False))
                    reached_any[aid] = bool(reached_any[aid] or reached)
                    unsafe_any[aid] = bool(unsafe_any[aid] or unsafe)

                    pos_new = np.asarray(next_states[aid].position, dtype=np.float32)
                    traj_len[aid] += float(np.linalg.norm(pos_new - last_pos[aid]))
                    last_pos[aid] = pos_new

                    if aid in active_agent_ids:
                        qp_total += 1
                        qp_feasible += 1 if bool(control_outputs[aid].solution.feasible) else 0

                safety_metrics = info.get("safety_metrics", {})
                for aid in agent_ids:
                    m = safety_metrics.get(aid, None)
                    if m is None:
                        continue
                    min_h_agent = min(min_h_agent, float(m.min_h_agent))
                    min_h_obs = min(min_h_obs, float(m.min_h_obstacle))

                obs = next_obs
                newly_frozen = {
                    aid for aid in active_agent_ids
                    if bool(info.get("reach_flags", {}).get(aid, False))
                }
                if newly_frozen and hasattr(self.env, "freeze_agents"):
                    self.env.freeze_agents(sorted(newly_frozen))
                frozen_agents.update(newly_frozen)

                if (len(switched_agents) > 0) and not (terminated or truncated):
                    sampled = _sample_high(obs)
                    states_round = {s.agent_id: s for s in self.env.get_agent_states()}
                    new_skills = {aid: int(sampled[aid]) for aid in switched_agents if aid not in frozen_agents}
                    if new_skills:
                        self.activate_round_skills(skill_map=new_skills, states=states_round)
                if terminated or truncated:
                    break

            n_agents = max(1, len(agent_ids))
            ep_stat = EvalEpisodeStats(
                episode_index=ep,
                success=bool(all(reached_any.values()) and not any(unsafe_any.values())),
                reach_rate=float(sum(1.0 for v in reached_any.values() if v) / n_agents),
                collision_rate=float(sum(1.0 for v in unsafe_any.values() if v) / n_agents),
                min_h_agent=float(min_h_agent if np.isfinite(min_h_agent) else 0.0),
                min_h_obstacle=float(min_h_obs if np.isfinite(min_h_obs) else 0.0),
                avg_traj_length=float(sum(traj_len.values()) / n_agents),
                skill_switches=int(skill_switches),
                qp_feasible_rate=float(qp_feasible / max(1, qp_total)),
                steps=int(len(trace_positions)),
                episode_return_mean=float(sum(ep_return.values()) / n_agents),
                safe_reach_ratio=float(
                    sum(1.0 for aid in agent_ids if reached_any[aid] and not unsafe_any[aid]) / n_agents
                ),
            )
            episode_stats.append(ep_stat)

            if bool(self.hooks.eval_render) and len(trace_positions) > 0:
                trace = EpisodeTrace(
                    positions=np.stack(trace_positions, axis=0),
                    goals=goals,
                    obstacles=obstacles,
                    unsafe_flags=np.stack(trace_unsafe, axis=0),
                    frame_labels=frame_labels,
                )
                out_dir = self.hooks.eval_render_dir
                if bool(self.hooks.eval_render_gif):
                    renderer.render_gif(
                        trace,
                        f"{out_dir}/episode_{ep:03d}.gif",
                        fps=max(1, int(self.hooks.eval_render_fps)),
                    )
                else:
                    renderer.render_static(trace, f"{out_dir}/episode_{ep:03d}.png")

        summary = evaluate_summary(episode_stats)
        return {
            "eval_n_episodes": float(summary.n_episodes),
            "eval_success_rate": float(summary.success_rate),
            "eval_reach_rate": float(summary.reach_rate),
            "eval_collision_rate": float(summary.collision_rate),
            "eval_safe_reach_ratio": float(summary.safe_reach_ratio),
            "eval_min_h_agent": float(summary.min_h_agent),
            "eval_min_h_obstacle": float(summary.min_h_obstacle),
            "eval_avg_traj_length": float(summary.avg_traj_length),
            "eval_avg_skill_switches": float(summary.avg_skill_switches),
            "eval_qp_feasible_rate": float(summary.qp_feasible_rate),
            "eval_avg_steps": float(summary.avg_steps),
            "eval_avg_episode_return": float(summary.avg_episode_return),
        }

    def train(self, total_iterations: int) -> None:
        for _ in range(total_iterations):
            self.collect_rollout()
            self.update_low_level()
            self.update_high_level()
