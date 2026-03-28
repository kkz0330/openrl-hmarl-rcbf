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
    low_update_epochs: int = 1
    low_max_samples_per_iter: int = 256
    low_target_step_scale: float = 0.05
    low_update_mode: str = "onpolicy_ppo"
    low_ppo_epochs: int = 2
    low_ppo_clip_ratio: float = 0.2
    low_ppo_value_coef: float = 0.5
    low_ppo_entropy_coef: float = 0.0
    low_ppo_max_grad_norm: float = 0.5
    low_policy_action_std: float = 0.20
    low_normalize_advantages: bool = True
    low_detach_value_head_in_actor: bool = True
    eval_episodes: int = 3
    eval_deterministic: bool = True
    eval_render: bool = False
    eval_render_dir: str = "artifacts/eval"
    eval_render_gif: bool = False


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
        self.hooks = hooks or TrainerHooks()

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
        return str(self.hooks.low_update_mode).strip().lower() in {"onpolicy_ppo", "low_ppo"}

    def _extract_local_context_from_info(
        self,
        tr: LowStepTransition,
    ) -> tuple[AgentState, List[AgentState], List[Dict[str, Any]], Dict[str, Any], np.ndarray]:
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
            d = dict(item)
            obstacles.append(
                {
                    "center": np.asarray(d["center"], dtype=np.float32).reshape(2),
                    "radius": float(d["radius"]),
                }
            )

        safety_constraints = dict(info.get("safety_constraints", {}))
        skill_u_ref = np.asarray(info.get("skill_u_ref", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
        return state, neighbors, obstacles, safety_constraints, skill_u_ref

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
            u_min=u_min,
            u_max=u_max,
        )
        return constants, u_min, u_max

    @staticmethod
    def _flatten_qp_param_torch(qp_param_raw: QPParam) -> QPParam:
        return QPParam(
            u_ref=qp_param_raw.u_ref.reshape(-1),
            r_diag=qp_param_raw.r_diag.reshape(-1),
            w_clf=qp_param_raw.w_clf.reshape(-1),
            cbf_k0=qp_param_raw.cbf_k0.reshape(-1),
            cbf_k1=qp_param_raw.cbf_k1.reshape(-1),
            clf_k=qp_param_raw.clf_k.reshape(-1),
            f_lin=None,
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
            u_ref=np.asarray(qp_param_raw.u_ref.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            r_diag=np.asarray(qp_param_raw.r_diag.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            w_clf=np.asarray(qp_param_raw.w_clf.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            cbf_k0=np.asarray(qp_param_raw.cbf_k0.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            cbf_k1=np.asarray(qp_param_raw.cbf_k1.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            clf_k=np.asarray(qp_param_raw.clf_k.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            f_lin=None,
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
        if self.diff_qp_solver is None:
            raise RuntimeError("differentiable QP solver is not set")
        if self.skill_runtime is None:
            raise RuntimeError("skill runtime manager is not set")
        if self.low_level_controller is None:
            raise RuntimeError("low level controller is not set")
        if self.low_policy is None or not hasattr(self.low_policy, "forward"):
            raise RuntimeError("low_policy must be a torch module for on-policy low-level updates")

        skill_targets = self.skill_runtime.control_targets(states=states, obs_low=obs_low, runtime_ctx=runtime_ctx)
        agent_ids = sorted(states.keys())
        std = max(1e-3, float(self.hooks.low_policy_action_std))

        actions: Dict[int, Any] = {}
        outputs: Dict[int, Any] = {}
        per_step_stats: Dict[int, Dict[str, Any]] = {}

        for aid in agent_ids:
            state_i = states[aid]
            target = dict(skill_targets[aid])
            skill_id = int(target["skill_id"])
            skill_u_ref = np.asarray(target["u_ref_skill"], dtype=np.float32).reshape(2)
            safety_constraints = dict(target.get("safety_constraints", {}))
            neighbors_all = [state_j for other_id, state_j in states.items() if other_id != aid]
            neighbors = self.low_level_controller._filter_neighbors(state_i=state_i, neighbors=neighbors_all)
            obstacles_local = self.low_level_controller._filter_obstacles(
                state_i=state_i,
                obstacles=obstacles,
                obs_low=obs_low[aid],
            )
            constants, _, _ = self._build_diff_constants(
                state_i=state_i,
                neighbors=neighbors,
                obstacles=obstacles_local,
                safety_constraints=safety_constraints,
            )

            obs_tensor = torch.as_tensor(obs_low[aid].flat, dtype=torch.float32).unsqueeze(0)
            skill_tensor = torch.as_tensor([skill_id], dtype=torch.long)
            with torch.no_grad():
                qp_param_raw = self.low_policy(obs_tensor, skill_tensor)
                qp_param = self._flatten_qp_param_torch(qp_param_raw)
                policy_u_ref = qp_param.u_ref.reshape(2)
                fused_mean_u_ref = self.low_level_controller._fuse_u_ref(
                    policy_u_ref=np.asarray(policy_u_ref.detach().cpu().numpy(), dtype=np.float32).reshape(2),
                    skill_u_ref=skill_u_ref,
                )
                qp_param_mean = QPParam(
                    u_ref=torch.as_tensor(fused_mean_u_ref, dtype=torch.float32),
                    r_diag=qp_param.r_diag,
                    w_clf=qp_param.w_clf,
                    cbf_k0=qp_param.cbf_k0,
                    cbf_k1=qp_param.cbf_k1,
                    clf_k=qp_param.clf_k,
                    f_lin=None,
                    hocbf_gamma_h=qp_param.hocbf_gamma_h,
                    hocbf_gamma_hdot=qp_param.hocbf_gamma_hdot,
                )
                mean_out = self.diff_qp_solver.solve(qp_param_mean, constants)
                mean_action = mean_out.action.reshape(2)
                noise = torch.randn_like(mean_action) * std
                sampled_proxy = mean_action + noise
                dist = torch.distributions.Normal(mean_action, torch.full_like(mean_action, std))
                logp = torch.sum(dist.log_prob(sampled_proxy))
                value = (
                    self.low_policy.low_value(obs_tensor, skill_tensor).reshape(1)
                    if hasattr(self.low_policy, "low_value")
                    else torch.zeros((1,), dtype=torch.float32)
                )
                qp_param_np = self._flatten_qp_param_numpy(qp_param_raw)
                sampled_proxy_np = np.asarray(sampled_proxy.detach().cpu().numpy(), dtype=np.float32).reshape(2)
                out = self.low_level_controller.solve_for_agent(
                    agent_id=aid,
                    state_i=state_i,
                    neighbors=neighbors,
                    obstacles=obstacles_local,
                    obs_low=obs_low[aid],
                    skill_id=skill_id,
                    skill_u_ref=skill_u_ref,
                    safety_constraints=safety_constraints,
                    qp_param_override=qp_param_np,
                    fused_u_ref_override=sampled_proxy_np,
                )
                actions[aid] = np.asarray(out.solution.action, dtype=np.float32).reshape(2)
                outputs[aid] = out
                per_step_stats[aid] = {
                    "low_logp": float(logp.detach().cpu().item()),
                    "low_value": float(value[0].detach().cpu().item()),
                    "low_policy_action_sample": sampled_proxy_np.copy(),
                    "low_policy_action_mean": np.asarray(mean_action.detach().cpu().numpy(), dtype=np.float32).reshape(2).copy(),
                    "skill_u_ref": skill_u_ref.copy(),
                    "low_policy_action_std": float(std),
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
            d_min_agent=d_min_agent,
            d_safe_obs=d_safe_obs,
            cbf_mode=str(overrides.get("cbf_mode", "distributed_ecbf")),
            cbf_u_max=float(overrides.get("cbf_u_max", max(np.max(np.abs(u_min)), np.max(np.abs(u_max)), 1e-3))),
            cbf_share_agent=float(overrides.get("cbf_share_agent", 0.5)),
            cbf_share_obs=float(overrides.get("cbf_share_obs", 1.0)),
            cbf_eps=float(overrides.get("cbf_eps", 1e-4)),
            u_min=u_min,
            u_max=u_max,
        )
        obs_tensor = torch.as_tensor(obs_low.flat, dtype=torch.float32).unsqueeze(0)
        skill_tensor = torch.as_tensor([skill_id], dtype=torch.long)
        qp_param_raw = self.low_policy(obs_tensor, skill_tensor)
        qp_param = QPParam(
            u_ref=qp_param_raw.u_ref.reshape(-1),
            r_diag=qp_param_raw.r_diag.reshape(-1),
            w_clf=qp_param_raw.w_clf.reshape(-1),
            cbf_k0=qp_param_raw.cbf_k0.reshape(-1),
            cbf_k1=qp_param_raw.cbf_k1.reshape(-1),
            clf_k=qp_param_raw.clf_k.reshape(-1),
            f_lin=None,
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

    def finalize_rollout_buffers(
        self,
        gamma_high: float = 0.99,
        lam_high: float = 0.95,
        gamma_low: float = 0.99,
        low_ext_reward_coef: float = 0.0,
        bootstrap_value_by_agent: Dict[int, float] | None = None,
    ) -> Dict[str, float]:
        high_stats = self.buffer.compute_high_advantages(
            gamma=gamma_high,
            lam=lam_high,
            use_gae=True,
            bootstrap_value_by_agent=bootstrap_value_by_agent,
        )
        low_stats = self.buffer.compute_low_returns(
            gamma=gamma_low,
            ext_reward_coef=low_ext_reward_coef,
            reset_on_sync_switch=True,
        )
        return {
            "high_n": high_stats["n_samples"],
            "high_adv_mean": high_stats["adv_mean"],
            "high_adv_std": high_stats["adv_std"],
            "low_n": low_stats["n_samples"],
            "low_return_mean": low_stats["return_mean"],
            "low_adv_mean": low_stats["adv_mean"],
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
        obs, _ = self.env.reset()
        agent_ids = sorted(obs.keys())
        self.skill_runtime.reset(agent_ids)
        self.coordinator.reset()

        ep_return_ext = {aid: 0.0 for aid in agent_ids}
        round_return_ext = {aid: 0.0 for aid in agent_ids}
        round_discount = {aid: 1.0 for aid in agent_ids}
        done_by_agent = {aid: False for aid in agent_ids}
        reached_any = {aid: False for aid in agent_ids}
        unsafe_any = {aid: False for aid in agent_ids}
        skill_counts_rollout = {int(sid): 0 for sid in sorted(self.skill_runtime.skill_by_id.keys())}
        last_obs = obs
        terminated = False
        truncated = False

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
                info={"round_start": True},
            )

        steps_collected = 0
        for _ in range(self.hooks.rollout_steps):
            states = {s.agent_id: s for s in self.env.get_agent_states()}
            obs_low = {aid: obs[aid]["low"] for aid in agent_ids}
            low_step_stats: Dict[int, Dict[str, Any]] = {}
            if self._is_low_update_ppo():
                actions, control_outputs, low_step_stats = self._compute_safe_actions_for_low_ppo(
                    states=states,
                    obs_low=obs_low,
                    obstacles=self.env.get_obstacles(),
                )
            else:
                actions, control_outputs = self.compute_safe_actions(
                    states=states,
                    obs_low=obs_low,
                    obstacles=self.env.get_obstacles(),
                )
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            next_states = {s.agent_id: s for s in self.env.get_agent_states()}
            next_obs_low = {aid: next_obs[aid]["low"] for aid in agent_ids}
            skill_out = self.skill_runtime.step_all(
                states=next_states,
                obs_low=next_obs_low,
                executed_actions=actions,
            )
            beta = {aid: bool(skill_out[aid].beta) for aid in agent_ids}
            step_sync = self.coordinator.step(beta)
            forced_end = bool(terminated or truncated)
            switched_agents = set(step_sync.switch_agents)
            if forced_end:
                switched_agents = set(agent_ids)

            for aid in agent_ids:
                unsafe = bool(info.get("unsafe_flags", {}).get(aid, False))
                reached = bool(info.get("reach_flags", {}).get(aid, False))
                unsafe_any[aid] = bool(unsafe_any[aid] or unsafe)
                reached_any[aid] = bool(reached_any[aid] or reached)
                done = bool(unsafe or reached or terminated or truncated)
                done_by_agent[aid] = done
                self.record_low_step(
                    LowStepTransition(
                        t=step_sync.t,
                        agent_id=aid,
                        obs_low=obs_low[aid],
                        skill_id=int(skill_out[aid].skill_id),
                        action=np.asarray(actions[aid], dtype=np.float32).reshape(2),
                        reward_int=float(skill_out[aid].intrinsic_reward),
                        reward_ext=float(rewards[aid]),
                        done=done,
                        logp=float(low_step_stats.get(aid, {}).get("low_logp")) if aid in low_step_stats else None,
                        value=float(low_step_stats.get(aid, {}).get("low_value")) if aid in low_step_stats else None,
                        sync_switch=bool(aid in switched_agents),
                        terminated_by_skill=bool(skill_out[aid].beta),
                        info={
                            "qp_feasible": bool(control_outputs[aid].solution.feasible),
                            "qp_status": control_outputs[aid].solution.solver_status,
                            "safety_constraints": dict(control_outputs[aid].safety_constraints),
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
                                {
                                    "center": np.asarray(o["center"], dtype=np.float32).reshape(2).copy(),
                                    "radius": float(o["radius"]),
                                }
                                for o in control_outputs[aid].obstacles_used
                            ],
                            "skill_u_ref": np.asarray(skill_out[aid].u_ref_skill, dtype=np.float32).reshape(2).copy(),
                            "low_policy_action_sample": (
                                np.asarray(low_step_stats[aid]["low_policy_action_sample"], dtype=np.float32).reshape(2).copy()
                                if aid in low_step_stats
                                else np.asarray(actions[aid], dtype=np.float32).reshape(2).copy()
                            ),
                            "low_policy_action_mean": (
                                np.asarray(low_step_stats[aid]["low_policy_action_mean"], dtype=np.float32).reshape(2).copy()
                                if aid in low_step_stats
                                else np.asarray(actions[aid], dtype=np.float32).reshape(2).copy()
                            ),
                            "low_policy_action_std": float(
                                low_step_stats.get(aid, {}).get("low_policy_action_std", self.hooks.low_policy_action_std)
                            ),
                        },
                    )
                )
                round_return_ext[aid] += round_discount[aid] * float(rewards[aid])
                round_discount[aid] *= self.hooks.gamma_high
                ep_return_ext[aid] += float(rewards[aid])

            steps_collected += 1
            last_obs = next_obs
            obs = next_obs

            if len(switched_agents) > 0:
                t_end = self.coordinator.t
                for aid in switched_agents:
                    if self.buffer.has_open_high_option(aid):
                        self.close_high_option(
                            agent_id=aid,
                            t_end=t_end,
                            return_ext=round_return_ext[aid],
                            done=done_by_agent[aid],
                            sync_switch=True,
                            info={"forced_sync": forced_end},
                        )
                if forced_end:
                    break

                sampled = _sample_high(obs)
                states_round = {s.agent_id: s for s in self.env.get_agent_states()}
                new_skills = {aid: int(sampled[aid]["skill_id"]) for aid in switched_agents}
                actual = self.activate_round_skills(
                    skill_map=new_skills,
                    states=states_round,
                )
                for aid in switched_agents:
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
                    )
                    round_return_ext[aid] = 0.0
                    round_discount[aid] = 1.0

        for aid in agent_ids:
            if self.buffer.has_open_high_option(aid):
                self.close_high_option(
                    agent_id=aid,
                    t_end=self.coordinator.t,
                    return_ext=round_return_ext[aid],
                    done=done_by_agent[aid],
                    sync_switch=True,
                    info={"cutoff_close": True},
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
        if not self._is_low_update_ppo():
            raise ValueError(
                "The paper-aligned subset only supports low_update_mode='onpolicy_ppo' or 'low_ppo'."
            )
        return self._update_low_level_onpolicy_ppo()

    def _update_low_level_target_regression(self) -> Dict[str, float]:
        if self.diff_qp_solver is None or self.low_level_optimizer is None:
            return {"n_updates": 0.0, "loss_mean": 0.0, "loss_last": 0.0}
        if self.low_policy is None or not hasattr(self.low_policy, "forward"):
            return {"n_updates": 0.0, "loss_mean": 0.0, "loss_last": 0.0}

        low_steps = self.buffer.snapshot()[0]
        if len(low_steps) == 0:
            return {"n_updates": 0.0, "loss_mean": 0.0, "loss_last": 0.0}

        selected = list(low_steps)[-self.hooks.low_max_samples_per_iter :]
        losses: List[float] = []
        n_updates = 0
        for _ in range(max(1, self.hooks.low_update_epochs)):
            for tr in selected:
                state, neighbors, obstacles, safety_constraints, _ = self._extract_local_context_from_info(tr)

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

                out = self.backward_low_level_diff_step(
                    state_i=state,
                    neighbors=neighbors,
                    obstacles=obstacles,
                    obs_low=tr.obs_low,
                    skill_id=int(tr.skill_id),
                    target_action=target_action,
                    optimizer=self.low_level_optimizer,
                    d_min_agent=float(self.constraint_builder.d_min_agent),
                    d_safe_obs=float(self.constraint_builder.d_safe_obs),
                    constraint_overrides=safety_constraints,
                )
                losses.append(float(out["loss"]))
                n_updates += 1

        return {
            "n_updates": float(n_updates),
            "loss_mean": float(sum(losses) / max(1, len(losses))),
            "loss_last": float(losses[-1] if losses else 0.0),
        }

    def _update_low_level_onpolicy_ppo(self) -> Dict[str, float]:
        if torch is None:
            raise RuntimeError("PyTorch is required for on-policy low-level updates")
        if self.diff_qp_solver is None or self.low_level_optimizer is None:
            return {"n_updates": 0.0, "loss_mean": 0.0, "loss_last": 0.0}
        if self.low_policy is None or not hasattr(self.low_policy, "forward") or not hasattr(self.low_policy, "low_value"):
            raise RuntimeError("low_policy must provide forward(...) and low_value(...) for on-policy low-level updates")
        if self.low_level_controller is None:
            raise RuntimeError("low level controller is not set")

        low_steps = self.buffer.snapshot()[0]
        if len(low_steps) == 0:
            return {"n_updates": 0.0, "loss_mean": 0.0, "loss_last": 0.0}

        selected = list(low_steps)[-self.hooks.low_max_samples_per_iter :]
        ppo_samples = [tr for tr in selected if tr.logp is not None and tr.return_target is not None]
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

        loss_all: List[float] = []
        loss_actor_all: List[float] = []
        loss_value_all: List[float] = []
        entropy_all: List[float] = []
        n_updates = 0

        n_epochs = max(1, int(self.hooks.low_ppo_epochs))
        for _ in range(n_epochs):
            order = np.random.permutation(len(ppo_samples))
            for idx in order:
                tr = ppo_samples[int(idx)]
                adv = float(adv_np[int(idx)])
                state, neighbors, obstacles, safety_constraints, skill_u_ref = self._extract_local_context_from_info(tr)
                constants, _, _ = self._build_diff_constants(
                    state_i=state,
                    neighbors=neighbors,
                    obstacles=obstacles,
                    safety_constraints=safety_constraints,
                )
                obs_tensor = torch.as_tensor(tr.obs_low.flat, dtype=torch.float32).unsqueeze(0)
                skill_tensor = torch.as_tensor([int(tr.skill_id)], dtype=torch.long)
                qp_param_raw = self.low_policy(obs_tensor, skill_tensor)
                qp_param = self._flatten_qp_param_torch(qp_param_raw)
                dtype = torch.float32
                device = constants.A_cbf.device
                skill_u_ref_t = torch.as_tensor(skill_u_ref, dtype=dtype, device=device).reshape(2)
                policy_u_ref_t = qp_param.u_ref.reshape(2).to(device=device, dtype=dtype)
                skill_ref_weight = float(self.low_level_controller.skill_ref_weight)
                fused_mean_u_ref = skill_ref_weight * skill_u_ref_t + (1.0 - skill_ref_weight) * policy_u_ref_t
                qp_param_mean = QPParam(
                    u_ref=fused_mean_u_ref,
                    r_diag=qp_param.r_diag,
                    w_clf=qp_param.w_clf,
                    cbf_k0=qp_param.cbf_k0,
                    cbf_k1=qp_param.cbf_k1,
                    clf_k=qp_param.clf_k,
                    f_lin=None,
                    hocbf_gamma_h=qp_param.hocbf_gamma_h,
                    hocbf_gamma_hdot=qp_param.hocbf_gamma_hdot,
                )
                mean_out = self.diff_qp_solver.solve(qp_param_mean, constants)
                mean_action = mean_out.action.reshape(2)
                sample_action = torch.as_tensor(
                    np.asarray(tr.info.get("low_policy_action_sample", tr.action), dtype=np.float32).reshape(2),
                    dtype=dtype,
                    device=device,
                )
                std = max(1e-3, float(tr.info.get("low_policy_action_std", self.hooks.low_policy_action_std)))
                dist = torch.distributions.Normal(mean_action, torch.full_like(mean_action, std))
                new_logp = torch.sum(dist.log_prob(sample_action))
                old_logp = torch.as_tensor(float(tr.logp), dtype=dtype, device=device)
                adv_t = torch.as_tensor(adv, dtype=dtype, device=device)
                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_t
                actor_loss = -torch.minimum(surr1, surr2)

                value_pred = self.low_policy.low_value(obs_tensor, skill_tensor).reshape(1).to(device=device, dtype=dtype)[0]
                ret_t = torch.as_tensor(float(tr.return_target), dtype=dtype, device=device)
                value_loss = 0.5 * (value_pred - ret_t) ** 2
                entropy_bonus = torch.sum(dist.entropy())

                total_loss = actor_loss + value_coef * value_loss - entropy_coef * entropy_bonus
                self.low_level_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.low_policy.parameters(), max_grad_norm)
                self.low_level_optimizer.step()

                loss_all.append(float(total_loss.detach().cpu().item()))
                loss_actor_all.append(float(actor_loss.detach().cpu().item()))
                loss_value_all.append(float(value_loss.detach().cpu().item()))
                entropy_all.append(float(entropy_bonus.detach().cpu().item()))
                n_updates += 1

        return {
            "n_updates": float(n_updates),
            "loss_mean": float(sum(loss_all) / max(1, len(loss_all))),
            "loss_last": float(loss_all[-1] if loss_all else 0.0),
            "loss_actor": float(sum(loss_actor_all) / max(1, len(loss_actor_all))),
            "loss_value": float(sum(loss_value_all) / max(1, len(loss_value_all))),
            "entropy": float(sum(entropy_all) / max(1, len(entropy_all))),
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

            reached_any = {aid: False for aid in agent_ids}
            unsafe_any = {aid: False for aid in agent_ids}
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
                actions, control_outputs = self.compute_safe_actions(
                    states=states,
                    obs_low=obs_low,
                    obstacles=self.env.get_obstacles(),
                )
                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                next_states = {s.agent_id: s for s in self.env.get_agent_states()}
                next_obs_low = {aid: next_obs[aid]["low"] for aid in agent_ids}
                skill_out = self.skill_runtime.step_all(
                    states=next_states,
                    obs_low=next_obs_low,
                    executed_actions=actions,
                )
                beta = {aid: bool(skill_out[aid].beta) for aid in agent_ids}
                sync_res = self.coordinator.step(beta)
                forced_end = bool(terminated or truncated)
                switched_agents = set(sync_res.switch_agents)
                if forced_end:
                    switched_agents = set(agent_ids)
                if len(switched_agents) > 0:
                    skill_switches += int(len(switched_agents))

                current_pos = np.stack([next_states[aid].position for aid in agent_ids], axis=0).astype(np.float32)
                trace_positions.append(current_pos)
                trace_unsafe.append(np.asarray([bool(info.get("unsafe_flags", {}).get(aid, False)) for aid in agent_ids], dtype=bool))

                for aid in agent_ids:
                    ep_return[aid] += float(rewards[aid])
                    reached = bool(info.get("reach_flags", {}).get(aid, False))
                    unsafe = bool(info.get("unsafe_flags", {}).get(aid, False))
                    reached_any[aid] = bool(reached_any[aid] or reached)
                    unsafe_any[aid] = bool(unsafe_any[aid] or unsafe)

                    pos_new = np.asarray(next_states[aid].position, dtype=np.float32)
                    traj_len[aid] += float(np.linalg.norm(pos_new - last_pos[aid]))
                    last_pos[aid] = pos_new

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
                if (len(switched_agents) > 0) and not (terminated or truncated):
                    sampled = _sample_high(obs)
                    states_round = {s.agent_id: s for s in self.env.get_agent_states()}
                    new_skills = {aid: int(sampled[aid]) for aid in switched_agents}
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
                )
                out_dir = self.hooks.eval_render_dir
                if bool(self.hooks.eval_render_gif):
                    renderer.render_gif(trace, f"{out_dir}/episode_{ep:03d}.gif")
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
