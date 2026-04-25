from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - import-safe fallback
    torch = None  # type: ignore[assignment]

from hmarl_cbf.baselines import (
    DistributedCBFBaselineConfig,
    DistributedCBFBaselineController,
)
from hmarl_cbf.control import ConstraintBuilder, DifferentiableQPSolver, LowLevelSafeController
from hmarl_cbf.env.obstacles import copy_obstacle, extract_lidar_cbf_obstacles
from hmarl_cbf.openrl_envs import LowLevelOpenRLEnv
from hmarl_cbf.policies import HighLevelPolicy, LowLevelQPPolicy
from hmarl_cbf.skills import build_default_skill_library
from hmarl_cbf.types import AgentObsLow, AgentState, QPSolution


@dataclass(slots=True)
class LowLevelTeacherDatasetSample:
    episode_index: int
    step_index: int
    agent_id: int
    actor_obs: np.ndarray
    critic_state: np.ndarray
    obs_low: AgentObsLow
    state: AgentState
    neighbors: List[AgentState]
    skill_id: int
    safety_constraints: Dict[str, Any]
    teacher_action: np.ndarray
    teacher_qp_feasible: bool
    teacher_solver_status: str
    teacher_qp_slack: np.ndarray
    teacher_qp_cbf_slack: np.ndarray
    reward_total: float
    terminated: bool
    truncated: bool


@dataclass(slots=True)
class HighLevelTeacherDatasetSample:
    episode_index: int
    step_index: int
    agent_id: int
    actor_obs: np.ndarray
    critic_state: np.ndarray
    action_mask: np.ndarray
    teacher_action: int


@dataclass(slots=True)
class LowLevelTeacherDataset:
    samples: List[LowLevelTeacherDatasetSample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            pickle.dump({"samples": self.samples, "metadata": self.metadata}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "LowLevelTeacherDataset":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        return cls(samples=list(payload["samples"]), metadata=dict(payload.get("metadata", {})))

    def __len__(self) -> int:
        return len(self.samples)


@dataclass(slots=True)
class TeacherDatasetCollectorConfig:
    episodes: int = 8
    max_steps_per_episode: int = 0
    seed: int = 0
    skill_selection_mode: str = "cyclic"
    require_teacher_feasible: bool = False


def _solver_used_fallback(status: str) -> bool:
    status_norm = str(status).strip().lower()
    return bool(status_norm.startswith("reduced_") or status_norm.startswith("stub_"))


def _load_state_dict_strict(module: Any, state: Mapping[str, Any], name: str) -> None:
    missing, unexpected = module.load_state_dict(dict(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{name} checkpoint load mismatch: missing={list(missing)} unexpected={list(unexpected)}"
        )


class TeacherCBFQPActionAdapter:
    """Low-level adapter that executes the hand-crafted teacher directly."""

    def __init__(self, controller: DistributedCBFBaselineController) -> None:
        self.controller = controller

    @staticmethod
    def _neighbor_payload(states: Mapping[int, AgentState], agent_id: int) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for other_id, state in states.items():
            if int(other_id) == int(agent_id):
                continue
            payload.append(
                {
                    "agent_id": int(state.agent_id),
                    "position": np.asarray(state.position, dtype=np.float32).reshape(2).copy(),
                    "velocity": np.asarray(state.velocity, dtype=np.float32).reshape(2).copy(),
                    "goal": np.asarray(state.goal, dtype=np.float32).reshape(2).copy(),
                    "radius": float(state.radius),
                }
            )
        return payload

    @staticmethod
    def _perceived_obstacles_payload(obs_low: AgentObsLow, controller: DistributedCBFBaselineController) -> List[Dict[str, Any]]:
        scan = obs_low.lidar_scan
        max_range = float(getattr(controller.config, "obstacle_range", scan.max_range))
        lidar_cfg = dict(getattr(controller.constraint_builder, "lidar_cbf_config", {}) or {})
        point_radius = float(lidar_cfg.get("point_radius", getattr(controller.config, "lidar_cbf_point_radius", 0.0)))
        top_k = int(lidar_cfg.get("top_k", getattr(controller.config, "lidar_cbf_top_k", 0)))
        perceived = extract_lidar_cbf_obstacles(
            scan=scan,
            max_range=max_range,
            use_fitted_geometry=bool(lidar_cfg.get("use_fitted_geometry", False)),
            point_radius=point_radius,
            top_k=top_k if top_k > 0 else None,
            min_segment_points=int(lidar_cfg.get("min_segment_points", 2)),
            line_fit_max_residual=float(lidar_cfg.get("line_fit_max_residual", 0.08)),
            circle_fit_max_residual=float(lidar_cfg.get("circle_fit_max_residual", 0.08)),
            circle_radius_min=float(lidar_cfg.get("circle_radius_min", 0.05)),
            circle_radius_max=float(lidar_cfg.get("circle_radius_max", 100.0)),
        )
        return [copy_obstacle(dict(obs)) for obs in perceived]

    def __call__(
        self,
        states: Mapping[int, AgentState],
        obs_low: Mapping[int, AgentObsLow],
        skill_targets: Mapping[int, Dict[str, Any]],
        phi_actions: Mapping[int, np.ndarray],
        core_env,
    ) -> tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        del phi_actions, skill_targets
        actions, solutions = self.controller.solve_batch(
            states=states,
            obstacles=core_env.get_obstacles(),
            obs_low=obs_low,
        )
        infos: Dict[int, Dict[str, Any]] = {}
        for agent_id, solution in solutions.items():
            cbf_slack = (
                np.asarray(solution.cbf_slack, dtype=np.float32).reshape(-1)
                if solution.cbf_slack is not None
                else np.zeros((0,), dtype=np.float32)
            )
            infos[int(agent_id)] = {
                "adapter_mode": "teacher_cbf_qp",
                "teacher_action": np.asarray(actions[int(agent_id)], dtype=np.float32).reshape(2).copy(),
                "executed_action": np.asarray(actions[int(agent_id)], dtype=np.float32).reshape(2).copy(),
                "teacher_qp_feasible": bool(solution.feasible),
                "teacher_solver_status": str(solution.solver_status),
                "teacher_qp_used_fallback": bool(_solver_used_fallback(str(solution.solver_status))),
                "teacher_qp_slack": np.asarray(solution.slack, dtype=np.float32).reshape(-1).copy(),
                "teacher_qp_cbf_slack": cbf_slack.copy(),
                "perceived_neighbors": self._neighbor_payload(states, int(agent_id)),
                "perceived_obstacles": self._perceived_obstacles_payload(obs_low[int(agent_id)], self.controller),
            }
        return (
            {int(agent_id): np.asarray(action, dtype=np.float32).reshape(2).copy() for agent_id, action in actions.items()},
            infos,
        )


class OriginalModelTeacher:
    """Teacher backed by a trained HMARL checkpoint.

    This uses the original high policy to choose skills at switch boundaries and the
    original low QP policy to produce skill-conditioned safe actions.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        runtime_cfg: Mapping[str, Any],
        torch_device: str = "cpu",
        deterministic: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to use a checkpoint-backed model teacher")
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"teacher checkpoint not found: {self.checkpoint_path}")
        self.runtime_cfg = dict(runtime_cfg)
        self.device = str(torch_device)
        self.deterministic = bool(deterministic)

        self._payload = torch.load(self.checkpoint_path, map_location=self.device)
        self._checkpoint_cfg = dict(self._payload.get("config", {})) if isinstance(self._payload, dict) else {}
        self._high_policy: HighLevelPolicy | None = None
        self._low_policy: LowLevelQPPolicy | None = None
        self._low_controller: LowLevelSafeController | None = None

    @staticmethod
    def _obs_dim_high_from_core(core_env) -> int:
        obs_high = core_env.get_obs_high()
        aid = sorted(obs_high.keys())[0]
        sample = obs_high[aid]
        return int(sample.self_state.shape[0] + sample.goal_relative.shape[0] + sample.neighbor_summary.shape[0])

    @staticmethod
    def _obs_dim_low_from_core(core_env) -> int:
        obs_low = core_env.get_obs_low()
        aid = sorted(obs_low.keys())[0]
        return int(obs_low[aid].flat.shape[0])

    def _effective_cfg(self) -> Mapping[str, Any]:
        if self._checkpoint_cfg:
            return self._checkpoint_cfg
        return self.runtime_cfg

    def _ensure_loaded(self, core_env) -> None:
        if self._high_policy is not None and self._low_policy is not None and self._low_controller is not None:
            return
        cfg = self._effective_cfg()
        obs_dim_high = self._obs_dim_high_from_core(core_env)
        obs_dim_low = self._obs_dim_low_from_core(core_env)
        n_skills = len(build_default_skill_library(max_duration=int(cfg["skills"]["default_max_duration"])))

        high_policy = HighLevelPolicy(
            obs_dim=obs_dim_high,
            n_skills=n_skills,
            hidden_dim=int(cfg["model"]["high_hidden_dim"]),
        ).to(self.device)
        low_policy = LowLevelQPPolicy(
            obs_dim=obs_dim_low,
            n_skills=n_skills,
            action_dim=int(cfg["model"]["action_dim"]),
            hidden_dim=int(cfg["model"]["low_hidden_dim"]),
            h_diag_min=float(cfg.get("low_level_qp", {}).get("h_diag_min", 1e-2)),
            h_diag_max=float(cfg.get("low_level_qp", {}).get("h_diag_max", 50.0)),
            h_offdiag_abs_max=float(cfg.get("low_level_qp", {}).get("h_offdiag_abs_max", 5.0)),
            f_abs_max=float(cfg.get("low_level_qp", {}).get("f_abs_max", 20.0)),
            phi_log_std_min=float(cfg.get("low_level_qp", {}).get("phi_log_std_min", -5.0)),
            phi_log_std_max=float(cfg.get("low_level_qp", {}).get("phi_log_std_max", 1.0)),
            w_clf=float(cfg.get("low_level_qp", {}).get("w_clf", 10.0)),
            w_cbf=float(cfg.get("low_level_qp", {}).get("w_cbf", 100.0)),
            cbf_slack_max=float(cfg.get("low_level_qp", {}).get("cbf_slack_max", 1.0)),
            cbf_k0=float(cfg.get("low_level_qp", {}).get("cbf_k0", 1.0)),
            cbf_k1=float(cfg.get("low_level_qp", {}).get("cbf_k1", 1.0)),
            clf_k=float(cfg.get("low_level_qp", {}).get("clf_k", 1.0)),
            hocbf_gamma_h=float(cfg.get("low_level_qp", {}).get("hocbf_gamma_h", 1.0)),
            hocbf_gamma_hdot=float(cfg.get("low_level_qp", {}).get("hocbf_gamma_hdot", 1.0)),
            f_residual_reference_enabled=bool(cfg.get("low_level_qp", {}).get("f_residual_reference_enabled", True)),
            f_ref_speed=float(cfg.get("low_level_qp", {}).get("f_ref_speed", cfg["skills"]["params"].get("ref_speed", 1.2))),
            f_ref_kp=float(cfg.get("low_level_qp", {}).get("f_ref_kp", 1.2)),
            f_ref_slow_radius=float(cfg.get("low_level_qp", {}).get("f_ref_slow_radius", cfg["skills"]["params"].get("slow_radius", 1.5))),
            f_ref_goal_stop_min_speed=float(cfg.get("low_level_qp", {}).get("f_ref_goal_stop_min_speed", cfg["skills"]["params"].get("goal_stop_min_speed", 0.0))),
        ).to(self.device)

        payload = dict(self._payload)
        if "high_policy" not in payload or "low_policy" not in payload:
            raise ValueError(f"checkpoint {self.checkpoint_path} does not contain high_policy and low_policy")
        _load_state_dict_strict(high_policy, payload["high_policy"], "high_policy")
        _load_state_dict_strict(low_policy, payload["low_policy"], "low_policy")
        high_policy.eval()
        low_policy.eval()

        runtime_cfg = self.runtime_cfg
        action_limit = float(runtime_cfg["env"]["action_limit"])
        constraint_builder = ConstraintBuilder(
            d_min_agent=float(runtime_cfg["safety"]["d_min_agent"]),
            d_safe_obs=float(runtime_cfg["safety"]["d_safe_obs"]),
            u_min=[-action_limit, -action_limit],
            u_max=[action_limit, action_limit],
            lidar_cbf_config={
                "use_fitted_geometry": bool(runtime_cfg["env"].get("lidar_cbf_use_fitted_geometry", False)),
                "point_radius": float(runtime_cfg["env"].get("lidar_cbf_point_radius", 0.0)),
                "top_k": int(runtime_cfg["env"].get("lidar_cbf_top_k", 3)),
                "min_segment_points": int(runtime_cfg["env"].get("lidar_cbf_min_segment_points", 2)),
                "line_fit_max_residual": float(runtime_cfg["env"].get("lidar_cbf_line_fit_max_residual", 0.08)),
                "circle_fit_max_residual": float(runtime_cfg["env"].get("lidar_cbf_circle_fit_max_residual", 0.08)),
                "circle_radius_min": float(runtime_cfg["env"].get("lidar_cbf_circle_radius_min", 0.05)),
                "circle_radius_max": float(runtime_cfg["env"].get("lidar_cbf_circle_radius_max", 100.0)),
            },
        )
        qp_solver = DifferentiableQPSolver(
            action_dim=int(runtime_cfg["model"]["action_dim"]),
            use_stub_if_unavailable=bool(runtime_cfg.get("qp", {}).get("use_stub_if_unavailable", False)),
            ecos_max_iters=int(runtime_cfg.get("qp", {}).get("ecos_max_iters", 500)),
            scs_max_iters=int(runtime_cfg.get("qp", {}).get("scs_max_iters", 10000)),
            scs_eps=float(runtime_cfg.get("qp", {}).get("scs_eps", 1e-4)),
        )
        self._low_controller = LowLevelSafeController(
            low_policy=low_policy,
            constraint_builder=constraint_builder,
            qp_solver=qp_solver,
            neighbor_perception_radius=float(runtime_cfg["env"].get("neighbor_radius", 0.0)),
            obstacle_perception_range=float(runtime_cfg["env"].get("lidar_range", 0.0)),
        )
        self._high_policy = high_policy
        self._low_policy = low_policy

    def select_skill_ids(self, low_env: LowLevelOpenRLEnv, agent_ids: Sequence[int]) -> Dict[int, int]:
        self._ensure_loaded(low_env.core_env)
        assert self._high_policy is not None
        actor_obs = low_env.core_env.build_high_actor_obs()
        ordered = [int(agent_id) for agent_id in agent_ids]
        batch = np.stack([np.asarray(actor_obs[aid], dtype=np.float32).reshape(-1) for aid in ordered], axis=0)
        with torch.no_grad():
            logits, _ = self._high_policy.forward(
                torch.as_tensor(batch, dtype=torch.float32, device=self.device),
            )
        skill_map: Dict[int, int] = {}
        for idx, agent_id in enumerate(ordered):
            available = list(low_env.get_available_skill_ids(int(agent_id)))
            if not available:
                raise RuntimeError(f"teacher high policy found no available skill for agent {agent_id}")
            row = logits[idx].detach().clone()
            legal_mask = torch.full_like(row, fill_value=False, dtype=torch.bool)
            legal_mask[torch.as_tensor(available, dtype=torch.long, device=row.device)] = True
            masked_row = row.masked_fill(~legal_mask, float("-inf"))
            if self.deterministic:
                chosen = int(torch.argmax(masked_row, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=masked_row)
                chosen = int(dist.sample().item())
            skill_map[int(agent_id)] = chosen
        return skill_map

    def solve_batch(
        self,
        *,
        states: Mapping[int, AgentState],
        obstacles: Sequence[Mapping[str, Any]],
        obs_low: Mapping[int, AgentObsLow],
        skill_targets: Mapping[int, Dict[str, Any]],
        core_env,
    ) -> tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        self._ensure_loaded(core_env)
        assert self._low_controller is not None
        outputs_actions, outputs = self._low_controller.solve_batch(
            states=states,
            obs_low=obs_low,
            skill_targets=skill_targets,
            obstacles=list(obstacles),
        )
        infos: Dict[int, Dict[str, Any]] = {}
        for agent_id, out in outputs.items():
            solution = out.solution
            infos[int(agent_id)] = {
                "adapter_mode": "teacher_original_model",
                "teacher_action": np.asarray(outputs_actions[int(agent_id)], dtype=np.float32).reshape(2).copy(),
                "executed_action": np.asarray(outputs_actions[int(agent_id)], dtype=np.float32).reshape(2).copy(),
                "teacher_qp_feasible": bool(solution.feasible),
                "teacher_solver_status": str(solution.solver_status),
                "teacher_qp_used_fallback": bool(_solver_used_fallback(str(solution.solver_status))),
                "teacher_qp_slack": np.asarray(solution.slack, dtype=np.float32).reshape(-1).copy(),
                "teacher_qp_cbf_slack": (
                    np.asarray(solution.cbf_slack, dtype=np.float32).reshape(-1).copy()
                    if solution.cbf_slack is not None
                    else np.zeros((0,), dtype=np.float32)
                ),
                "perceived_neighbors": TeacherCBFQPActionAdapter._neighbor_payload(states, int(agent_id)),
                "perceived_obstacles": [copy_obstacle(dict(obs)) for obs in out.obstacles_used],
            }
        return (
            {int(agent_id): np.asarray(action, dtype=np.float32).reshape(2).copy() for agent_id, action in outputs_actions.items()},
            infos,
        )


class TeacherDatasetCollector:
    def __init__(
        self,
        env: LowLevelOpenRLEnv,
        teacher_controller: Any,
        config: TeacherDatasetCollectorConfig | None = None,
    ) -> None:
        self.env = env
        self.teacher_controller = teacher_controller
        self.config = config or TeacherDatasetCollectorConfig()
        self._rng = np.random.default_rng(int(self.config.seed))
        if hasattr(teacher_controller, "solve_batch") and hasattr(teacher_controller, "select_skill_ids"):
            self._teacher_adapter = self
        else:
            self._teacher_adapter = TeacherCBFQPActionAdapter(teacher_controller)
        self._skill_cycle_cursor: Dict[int, int] = {int(agent_id): 0 for agent_id in self.env.possible_agents}
        self.last_high_level_samples: List[HighLevelTeacherDatasetSample] = []

    @staticmethod
    def _copy_state(state: AgentState) -> AgentState:
        return AgentState(
            agent_id=int(state.agent_id),
            position=np.asarray(state.position, dtype=np.float32).reshape(2).copy(),
            velocity=np.asarray(state.velocity, dtype=np.float32).reshape(2).copy(),
            goal=np.asarray(state.goal, dtype=np.float32).reshape(2).copy(),
            radius=float(state.radius),
        )

    @staticmethod
    def _copy_obs_low(obs_low: AgentObsLow) -> AgentObsLow:
        scan = obs_low.lidar_scan
        from hmarl_cbf.types import LidarScan

        return AgentObsLow(
            self_state=np.asarray(obs_low.self_state, dtype=np.float32).copy(),
            goal_relative=np.asarray(obs_low.goal_relative, dtype=np.float32).reshape(2).copy(),
            neighbor_summary=np.asarray(obs_low.neighbor_summary, dtype=np.float32).copy(),
            lidar_scan=LidarScan(
                ranges=np.asarray(scan.ranges, dtype=np.float32).copy(),
                max_range=float(scan.max_range),
                angles=np.asarray(scan.angles, dtype=np.float32).copy(),
                origin=np.asarray(scan.origin, dtype=np.float32).reshape(2).copy(),
                hit_points=np.asarray(scan.hit_points, dtype=np.float32).copy(),
                hit_valid=np.asarray(scan.hit_valid, dtype=np.bool_).copy(),
                hit_kinds=np.asarray(scan.hit_kinds, dtype=np.int32).copy(),
                noise_std=float(scan.noise_std),
            ),
        )

    def _select_skill_for_agent(self, agent_id: int) -> int:
        available = self.env.get_available_skill_ids(int(agent_id))
        if not available:
            raise RuntimeError(f"agent {agent_id} has no available skill for teacher collection")
        mode = str(self.config.skill_selection_mode).strip().lower()
        if mode == "random":
            return int(self._rng.choice(np.asarray(available, dtype=np.int64)))
        if mode == "cyclic":
            cursor = int(self._skill_cycle_cursor[int(agent_id)]) % len(available)
            self._skill_cycle_cursor[int(agent_id)] = cursor + 1
            return int(available[cursor])
        raise ValueError(f"unknown skill_selection_mode: {self.config.skill_selection_mode}")

    def _assign_pending_skills(self) -> None:
        if not self.env.has_pending_switch():
            return
        episode_index = getattr(self, "_active_episode_index", 0)
        step_index = getattr(self, "_active_step_index", 0)
        pending_agents = list(self.env.pending_switch_agents)
        high_actor_obs = self.env.core_env.build_high_actor_obs()
        high_critic_state = self.env.core_env.build_high_critic_obs()
        if hasattr(self.teacher_controller, "select_skill_ids"):
            skill_map = self.teacher_controller.select_skill_ids(self.env, self.env.pending_switch_agents)
        else:
            skill_map = {
                int(agent_id): self._select_skill_for_agent(int(agent_id))
                for agent_id in self.env.pending_switch_agents
            }
        for agent_id in pending_agents:
            available = self.env.get_available_skill_ids(int(agent_id))
            action_mask = np.zeros((int(self.env.config.n_skills),), dtype=np.float32)
            for skill_id in available:
                if 0 <= int(skill_id) < int(self.env.config.n_skills):
                    action_mask[int(skill_id)] = 1.0
            self.last_high_level_samples.append(
                HighLevelTeacherDatasetSample(
                    episode_index=int(episode_index),
                    step_index=int(step_index),
                    agent_id=int(agent_id),
                    actor_obs=np.asarray(high_actor_obs[int(agent_id)], dtype=np.float32).reshape(-1).copy(),
                    critic_state=np.asarray(high_critic_state, dtype=np.float32).reshape(-1).copy(),
                    action_mask=action_mask,
                    teacher_action=int(skill_map[int(agent_id)]),
                )
            )
        self.env.set_skill_map(skill_map, validate=True, require_all=False)

    def __call__(
        self,
        states: Mapping[int, AgentState],
        obs_low: Mapping[int, AgentObsLow],
        skill_targets: Mapping[int, Dict[str, Any]],
        phi_actions: Mapping[int, np.ndarray],
        core_env,
    ) -> tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        del phi_actions
        return self.teacher_controller.solve_batch(
            states=states,
            obstacles=core_env.get_obstacles(),
            obs_low=obs_low,
            skill_targets=skill_targets,
            core_env=core_env,
        )

    def collect(self) -> LowLevelTeacherDataset:
        original_adapter = self.env.action_adapter
        self.env.action_adapter = self._teacher_adapter
        samples: List[LowLevelTeacherDatasetSample] = []
        self.last_high_level_samples = []
        try:
            for episode_index in range(max(1, int(self.config.episodes))):
                self._active_episode_index = int(episode_index)
                seed = int(self.config.seed + episode_index)
                print("teacher_collect_episode_start", int(episode_index + 1))
                obs, infos = self.env.reset(seed=seed)
                del obs, infos
                step_limit = (
                    int(self.config.max_steps_per_episode)
                    if int(self.config.max_steps_per_episode) > 0
                    else int(self.env.core_env.env.horizon)
                )

                for step_index in range(step_limit):
                    self._active_step_index = int(step_index)
                    self._assign_pending_skills()
                    if not self.env.agents:
                        break

                    actor_obs = self.env.current_actor_obs()
                    infos = self.env.current_infos()
                    states = self.env.core_env.get_states()
                    obs_low = self.env.core_env.get_obs_low()
                    start_agents = list(self.env.agents)
                    phi_zeros = {
                        int(agent_id): np.zeros((int(self.env.phi_dim),), dtype=np.float32)
                        for agent_id in start_agents
                    }

                    next_obs, rewards, terminations, truncations, next_infos = self.env.step(phi_zeros)
                    del next_obs

                    for agent_id in start_agents:
                        step_info = dict(next_infos[int(agent_id)])
                        sample = LowLevelTeacherDatasetSample(
                            episode_index=int(episode_index),
                            step_index=int(step_index),
                            agent_id=int(agent_id),
                            actor_obs=np.asarray(actor_obs[int(agent_id)], dtype=np.float32).reshape(-1).copy(),
                            critic_state=np.asarray(infos[int(agent_id)]["critic_state"], dtype=np.float32).reshape(-1).copy(),
                            obs_low=self._copy_obs_low(obs_low[int(agent_id)]),
                            state=self._copy_state(states[int(agent_id)]),
                            neighbors=[
                                self._copy_state(states[int(other_id)])
                                for other_id in start_agents
                                if int(other_id) != int(agent_id)
                            ],
                            skill_id=int(infos[int(agent_id)]["skill_id"]),
                            safety_constraints=dict(step_info.get("safety_constraints", {})),
                            teacher_action=np.asarray(
                                step_info.get("teacher_action", step_info.get("executed_action")),
                                dtype=np.float32,
                            ).reshape(2).copy(),
                            teacher_qp_feasible=bool(step_info.get("teacher_qp_feasible", True)),
                            teacher_solver_status=str(step_info.get("teacher_solver_status", "unknown")),
                            teacher_qp_slack=np.asarray(
                                step_info.get("teacher_qp_slack", np.zeros((1,), dtype=np.float32)),
                                dtype=np.float32,
                            ).reshape(-1).copy(),
                            teacher_qp_cbf_slack=np.asarray(
                                step_info.get("teacher_qp_cbf_slack", np.zeros((0,), dtype=np.float32)),
                                dtype=np.float32,
                            ).reshape(-1).copy(),
                            reward_total=float(rewards[int(agent_id)]),
                            terminated=bool(terminations[int(agent_id)]),
                            truncated=bool(truncations[int(agent_id)]),
                        )
                        if self.config.require_teacher_feasible and not sample.teacher_qp_feasible:
                            continue
                        samples.append(sample)

                    if not self.env.agents or all(bool(terminations[aid] or truncations[aid]) for aid in start_agents):
                        break
                print("teacher_collect_episode_samples", int(len(samples)))
        finally:
            self.env.action_adapter = original_adapter
            self._active_episode_index = 0
            self._active_step_index = 0

        metadata = {
            "episodes": int(self.config.episodes),
            "seed": int(self.config.seed),
            "skill_selection_mode": str(self.config.skill_selection_mode),
            "sample_count": int(len(samples)),
            "high_sample_count": int(len(self.last_high_level_samples)),
            "n_skills": int(self.env.config.n_skills),
        }
        return LowLevelTeacherDataset(samples=samples, metadata=metadata)


def build_teacher_baseline_controller_from_config(cfg: Mapping[str, Any]) -> DistributedCBFBaselineController:
    action_limit = float(cfg["env"]["action_limit"])
    teacher_cfg_raw = dict(cfg.get("teacher_baseline", {}))
    gcbf_style_cfg = dict(cfg.get("gcbf_style_handcrafted_baseline", {}))
    agent_radius = float(cfg["env"].get("agent_radius", 0.05))
    gcbf_safe_distance = float(max(1e-4, 2.0 * agent_radius))
    teacher_d_min_agent = float(
        teacher_cfg_raw.get(
            "d_min_agent",
            gcbf_style_cfg.get("d_min_agent", gcbf_safe_distance),
        )
    )
    teacher_d_safe_obs = float(
        teacher_cfg_raw.get(
            "d_safe_obs",
            gcbf_style_cfg.get("d_safe_obs", gcbf_safe_distance),
        )
    )
    teacher_point_radius = float(
        teacher_cfg_raw.get(
            "lidar_cbf_point_radius",
            gcbf_style_cfg.get("lidar_cbf_point_radius", cfg["env"].get("lidar_cbf_point_radius", 0.0)),
        )
    )
    teacher_top_k = int(
        teacher_cfg_raw.get(
            "lidar_cbf_top_k",
            gcbf_style_cfg.get("k", cfg["env"].get("lidar_cbf_top_k", 3)),
        )
    )
    constraint_builder = ConstraintBuilder(
        d_min_agent=teacher_d_min_agent,
        d_safe_obs=teacher_d_safe_obs,
        u_min=[-action_limit, -action_limit],
        u_max=[action_limit, action_limit],
        lidar_cbf_config={
            "use_fitted_geometry": bool(cfg["env"].get("lidar_cbf_use_fitted_geometry", False)),
            "point_radius": teacher_point_radius,
            "top_k": teacher_top_k,
            "min_segment_points": int(cfg["env"].get("lidar_cbf_min_segment_points", 2)),
            "line_fit_max_residual": float(cfg["env"].get("lidar_cbf_line_fit_max_residual", 0.08)),
            "circle_fit_max_residual": float(cfg["env"].get("lidar_cbf_circle_fit_max_residual", 0.08)),
            "circle_radius_min": float(cfg["env"].get("lidar_cbf_circle_radius_min", 0.05)),
            "circle_radius_max": float(cfg["env"].get("lidar_cbf_circle_radius_max", 100.0)),
        },
    )
    qp_solver = DifferentiableQPSolver(
        action_dim=int(cfg["model"]["action_dim"]),
        use_stub_if_unavailable=bool(cfg.get("qp", {}).get("use_stub_if_unavailable", False)),
        ecos_max_iters=int(cfg.get("qp", {}).get("ecos_max_iters", 500)),
        scs_max_iters=int(cfg.get("qp", {}).get("scs_max_iters", 10000)),
        scs_eps=float(cfg.get("qp", {}).get("scs_eps", 1e-4)),
    )
    teacher_cfg = DistributedCBFBaselineConfig.from_mapping(
        {
            "nominal_mode": teacher_cfg_raw.get("nominal_mode", "gcbf_lqr"),
            "use_clf": teacher_cfg_raw.get("use_clf", False),
            "cbf_mode": cfg["skills"]["params"].get("cbf_mode", "distributed_gcbfplus"),
            "cbf_share_agent": cfg["skills"]["params"].get("cbf_share_agent", 0.5),
            "cbf_share_obs": cfg["skills"]["params"].get("cbf_share_obs", 1.0),
            "cbf_u_max": action_limit,
            "robust_cbf": cfg.get("safety", {}).get("robust_cbf", False),
            "disturbance_accel_max": cfg["env"].get("disturbance_accel_max", 0.0),
            "relative_disturbance_accel_max": cfg.get("safety", {}).get("relative_disturbance_accel_max", 0.0),
            "cbf_k0": cfg.get("low_level_qp", {}).get("cbf_k0", 1.0),
            "cbf_k1": cfg.get("low_level_qp", {}).get("cbf_k1", 1.0),
            "hocbf_gamma_h": cfg.get("low_level_qp", {}).get("hocbf_gamma_h", 1.0),
            "hocbf_gamma_hdot": cfg.get("low_level_qp", {}).get("hocbf_gamma_hdot", 1.0),
            "clf_k": cfg.get("low_level_qp", {}).get("clf_k", 1.0),
            "H_diag": [1.0, 1.0],
            "w_clf": 0.0,
            "w_cbf": teacher_cfg_raw.get(
                "relax_penalty",
                gcbf_style_cfg.get("relax_penalty", cfg.get("low_level_qp", {}).get("w_cbf", 100.0)),
            ),
            "cbf_slack_max": teacher_cfg_raw.get(
                "cbf_slack_max",
                gcbf_style_cfg.get("cbf_slack_max", 1.0e6),
            ),
            "ref_speed": cfg["skills"]["params"].get("ref_speed", 1.2),
            "speed_kp": teacher_cfg_raw.get("speed_kp", 1.2),
            "slow_radius": cfg["skills"]["params"].get("slow_radius", 1.5),
            "goal_stop_min_speed": cfg["skills"]["params"].get("goal_stop_min_speed", 0.0),
            "lqr_q_pos": teacher_cfg_raw.get("lqr_q_pos", gcbf_style_cfg.get("q_pos", 5.0)),
            "lqr_q_vel": teacher_cfg_raw.get("lqr_q_vel", gcbf_style_cfg.get("q_vel", 5.0)),
            "lqr_r_input": teacher_cfg_raw.get("lqr_r_input", gcbf_style_cfg.get("r_input", 1.0)),
            "lqr_error_clip_radius": teacher_cfg_raw.get(
                "lqr_error_clip_radius", cfg["env"].get("lidar_range", 3.0)
            ),
            "gcbf_mass": teacher_cfg_raw.get("gcbf_mass", gcbf_style_cfg.get("mass", 1.0)),
            "gcbf_comm_radius": teacher_cfg_raw.get(
                "gcbf_comm_radius",
                gcbf_style_cfg.get("comm_radius", cfg["env"].get("neighbor_radius", 3.0)),
            ),
            "dt": teacher_cfg_raw.get(
                "dt",
                cfg["env"].get("dt", gcbf_style_cfg.get("dt", 0.03)),
            ),
            "velocity_limit": teacher_cfg_raw.get(
                "velocity_limit",
                gcbf_style_cfg.get("velocity_limit", cfg["env"].get("velocity_limit", 2.0)),
            ),
            "mpc_horizon": teacher_cfg_raw.get("mpc_horizon", 12),
            "mpc_q_pos": teacher_cfg_raw.get("mpc_q_pos", 6.0),
            "mpc_q_vel": teacher_cfg_raw.get("mpc_q_vel", 0.8),
            "mpc_q_terminal_pos": teacher_cfg_raw.get("mpc_q_terminal_pos", 10.0),
            "mpc_q_terminal_vel": teacher_cfg_raw.get("mpc_q_terminal_vel", 1.0),
            "mpc_r_input": teacher_cfg_raw.get("mpc_r_input", 0.15),
            "mpc_obs_extra_margin": teacher_cfg_raw.get("mpc_obs_extra_margin", 0.0),
            "mpc_obs_constraint_horizon": teacher_cfg_raw.get("mpc_obs_constraint_horizon", 6),
            "neighbor_radius": teacher_cfg_raw.get(
                "neighbor_radius",
                gcbf_style_cfg.get("comm_radius", cfg["env"].get("neighbor_radius", 2.0)),
            ),
            "obstacle_range": teacher_cfg_raw.get(
                "obstacle_range",
                gcbf_style_cfg.get("comm_radius", cfg["env"].get("lidar_range", 3.0)),
            ),
            "prefilter_neighbors": teacher_cfg_raw.get("prefilter_neighbors", False),
            "prefilter_obstacles_top_k": teacher_cfg_raw.get("prefilter_obstacles_top_k", False),
            "boundary_cbf": teacher_cfg_raw.get("boundary_cbf", cfg.get("safety", {}).get("boundary_cbf", True)),
            "boundary_margin": teacher_cfg_raw.get(
                "boundary_margin",
                cfg.get("safety", {}).get("boundary_margin", cfg["env"].get("agent_radius", 0.05)),
            ),
            "world_size": cfg["env"].get("world_size", 10.0),
            "rect_base_margin_extra": cfg["env"].get("rect_base_margin_extra", 0.0),
            "rect_corner_margin_enabled": cfg["env"].get("rect_corner_margin_enabled", False),
            "rect_corner_margin_max": cfg["env"].get("rect_corner_margin_max", 0.0),
            "rect_corner_proximity_distance": cfg["env"].get("rect_corner_proximity_distance", 0.4),
            "rect_corner_speed_min": cfg["env"].get("rect_corner_speed_min", 0.05),
            "rect_corner_alignment_power": cfg["env"].get("rect_corner_alignment_power", 1.0),
            "rect_dual_edge_cbf_enabled": cfg["env"].get("rect_dual_edge_cbf_enabled", False),
            "rect_dual_edge_proximity_distance": cfg["env"].get("rect_dual_edge_proximity_distance", 0.0),
            "rect_smooth_tau": cfg["env"].get("rect_smooth_tau", 0.1),
            "lidar_cbf_use_fitted_geometry": cfg["env"].get("lidar_cbf_use_fitted_geometry", False),
            "lidar_cbf_point_radius": teacher_point_radius,
            "lidar_cbf_top_k": teacher_top_k,
            "lidar_cbf_min_segment_points": cfg["env"].get("lidar_cbf_min_segment_points", 2),
            "lidar_cbf_line_fit_max_residual": cfg["env"].get("lidar_cbf_line_fit_max_residual", 0.08),
            "lidar_cbf_circle_fit_max_residual": cfg["env"].get("lidar_cbf_circle_fit_max_residual", 0.08),
            "lidar_cbf_circle_radius_min": cfg["env"].get("lidar_cbf_circle_radius_min", 0.05),
            "lidar_cbf_circle_radius_max": cfg["env"].get("lidar_cbf_circle_radius_max", 100.0),
            "use_input_bounds": True,
        }
    )
    return DistributedCBFBaselineController(
        constraint_builder=constraint_builder,
        qp_solver=qp_solver,
        config=teacher_cfg,
    )


def build_original_model_teacher_from_checkpoint(
    cfg: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    torch_device: str = "cpu",
    deterministic: bool = True,
) -> OriginalModelTeacher:
    return OriginalModelTeacher(
        checkpoint_path=checkpoint_path,
        runtime_cfg=cfg,
        torch_device=torch_device,
        deterministic=deterministic,
    )


def build_teacher_controller_from_config(
    cfg: Mapping[str, Any],
    *,
    teacher_source: str = "baseline",
    teacher_checkpoint: str | Path = "",
    torch_device: str = "cpu",
    deterministic: bool = True,
) -> Any:
    source = str(teacher_source).strip().lower()
    if source == "baseline":
        return build_teacher_baseline_controller_from_config(cfg)
    if source == "model_checkpoint":
        if not str(teacher_checkpoint).strip():
            raise ValueError("teacher_checkpoint is required when teacher_source=model_checkpoint")
        return build_original_model_teacher_from_checkpoint(
            cfg,
            checkpoint_path=teacher_checkpoint,
            torch_device=torch_device,
            deterministic=deterministic,
        )
    raise ValueError(f"unknown teacher_source: {teacher_source}")
