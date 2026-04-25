from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

from hmarl_cbf.openrl_envs.core_env import CoreEnv
from hmarl_cbf.openrl_agents import TorchDiffQPActionAdapter
from hmarl_cbf.openrl_models.low_qp_decoder import LowQPDecoder
from hmarl_cbf.control import ConstraintBuilder, TorchDifferentiableQPSolver
from hmarl_cbf.skills import SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.types import AgentObsLow, AgentState, SkillSpec


LowLevelActionAdapter = Callable[
    [
        Mapping[int, AgentState],
        Mapping[int, AgentObsLow],
        Mapping[int, Dict[str, Any]],
        Mapping[int, np.ndarray],
        CoreEnv,
    ],
    tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]]],
]


@dataclass(slots=True)
class LowLevelOpenRLEnvConfig:
    n_skills: int
    phi_dim: int = 5
    freeze_on_reach: bool = True
    include_local_obs_in_critic: bool = False
    intrinsic_reward_coef: float = 0.0
    reactivate_same_skill_on_switch: bool = False
    default_skill_id: int = 0
    d_min_agent: float = 0.6
    d_safe_obs: float = 0.6
    neighbor_perception_radius: float | None = None
    obstacle_perception_range: float | None = None
    torch_device: str = "cpu"
    decoder_config: Dict[str, Any] = field(default_factory=dict)
    solver_config: Dict[str, Any] = field(default_factory=dict)
    skill_context: Dict[str, Any] = field(default_factory=dict)


class LowLevelOpenRLEnv:
    """OpenRL-friendly low-level environment view.

    One low-level step:
    - consumes per-agent continuous ``phi`` vectors
    - decodes them into QP parameters
    - solves the torch diff-QP safety layer
    - executes the resulting safe action
    - advances the underlying physics by one step
    - updates per-agent skill runtime and returns low-level observations/rewards
    """

    metadata = {"name": "hmarl_cbf_low_level_openrl"}

    def __init__(
        self,
        core_env: CoreEnv,
        config: LowLevelOpenRLEnvConfig,
        *,
        skill_library: Iterable[SkillSpec] | None = None,
        action_adapter: LowLevelActionAdapter | None = None,
    ) -> None:
        self.core_env = core_env
        self.config = config
        self.possible_agents = list(range(int(self.core_env.env.n_agents)))
        self.agents = list(self.possible_agents)
        self.agent_num = len(self.possible_agents)
        self.parallel_env_num = 1
        self.env_name = str(self.metadata["name"])

        skills = list(skill_library) if skill_library is not None else build_default_skill_library()
        self.skill_runtime = SkillRuntimeManager(skills=skills, default_ctx=self._default_skill_context())
        self.action_adapter = action_adapter if action_adapter is not None else self._build_default_action_adapter()
        decoder_kwargs = dict(self.config.decoder_config)
        decoder_action_dim = int(decoder_kwargs.get("action_dim", 2))
        decoder_parameterize_cbf = bool(decoder_kwargs.get("parameterize_cbf_constraints", False))
        derived_phi_dim = LowQPDecoder.compute_phi_dim(
            decoder_action_dim,
            parameterize_cbf_constraints=decoder_parameterize_cbf,
        )
        requested_phi_dim = int(self.config.phi_dim)
        if requested_phi_dim > 0 and requested_phi_dim != derived_phi_dim:
            raise ValueError(
                f"LowLevelOpenRLEnvConfig.phi_dim={requested_phi_dim} does not match decoder-derived "
                f"phi_dim={derived_phi_dim}"
            )
        self.phi_dim = derived_phi_dim if requested_phi_dim <= 0 else requested_phi_dim

        lidar_rays = int(self.core_env.env.lidar.n_beam)
        max_neighbors = int(self.core_env.env.obs_builder.max_neighbors)
        self._actor_obs_dim = int(4 + 2 + lidar_rays + max_neighbors * 4 + int(self.config.n_skills))
        self._critic_obs_dim = int(
            self.core_env.build_low_critic_obs(
                skill_ids={aid: 0 for aid in self.possible_agents},
                n_skills=self.config.n_skills,
                include_local_low_obs=bool(self.config.include_local_obs_in_critic),
            ).shape[0]
        )

        if spaces is not None:
            self.observation_spaces = {
                agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(self._actor_obs_dim,), dtype=np.float32)
                for agent_id in self.possible_agents
            }
            self.action_spaces = {
                agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(int(self.phi_dim),), dtype=np.float32)
                for agent_id in self.possible_agents
            }
            self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._critic_obs_dim,), dtype=np.float32)
            self.observation_space = self.observation_spaces[self.possible_agents[0]]
            self.action_space = self.action_spaces[self.possible_agents[0]]
        else:
            self.observation_spaces = {}
            self.action_spaces = {}
            self.state_space = None
            self.observation_space = None
            self.action_space = None

        self._skill_ids: Dict[int, int] = {}
        self._pending_reactivate: set[int] = set()
        self._done_agents: set[int] = set()
        self._frozen_agents: set[int] = set()
        self._last_skill_outputs: Dict[int, Dict[str, Any]] = {}
        self._last_adapter_info: Dict[int, Dict[str, Any]] = {}

    def _default_skill_context(self) -> Dict[str, Any]:
        env = self.core_env.env
        ctx = {
            "action_limit": float(env.action_limit),
            "u_min": np.asarray([-env.action_limit, -env.action_limit], dtype=np.float32),
            "u_max": np.asarray([env.action_limit, env.action_limit], dtype=np.float32),
            "velocity_limit": float(env.velocity_limit),
            "goal_threshold": float(env.goal_threshold),
            "goal_speed_threshold": float(env.goal_speed_threshold),
            "slow_radius": 1.5,
            "goal_stop_min_speed": 0.0,
            "world_size": float(env.world_size),
            "boundary_cbf": True,
            "boundary_margin": float(env.agent_radius),
            "cbf_mode": "distributed_gcbfplus",
            "d_min_agent": float(self.config.d_min_agent),
            "d_safe_obs": float(self.config.d_safe_obs),
            "obstacle_perception_range": float(
                self.config.obstacle_perception_range
                if self.config.obstacle_perception_range is not None
                else env.lidar.max_range
            ),
            "lidar_obstacle_cbf_enabled": bool(getattr(env, "lidar_obstacle_cbf_enabled", True)),
            "lidar_cbf_use_fitted_geometry": bool(getattr(env, "lidar_cbf_use_fitted_geometry", False)),
            "lidar_cbf_point_radius": float(getattr(env, "lidar_cbf_point_radius", 0.0)),
            "lidar_cbf_top_k": int(getattr(env, "lidar_cbf_top_k", 3)),
            "lidar_cbf_min_segment_points": int(getattr(env, "lidar_cbf_min_segment_points", 2)),
            "lidar_cbf_line_fit_max_residual": float(getattr(env, "lidar_cbf_line_fit_max_residual", 0.08)),
            "lidar_cbf_circle_fit_max_residual": float(getattr(env, "lidar_cbf_circle_fit_max_residual", 0.08)),
            "lidar_cbf_circle_radius_min": float(getattr(env, "lidar_cbf_circle_radius_min", 0.05)),
            "lidar_cbf_circle_radius_max": float(getattr(env, "lidar_cbf_circle_radius_max", 100.0)),
            "rect_base_margin_extra": float(getattr(env, "rect_base_margin_extra", 0.0)),
            "rect_corner_margin_enabled": bool(getattr(env, "rect_corner_margin_enabled", False)),
            "rect_corner_margin_max": float(getattr(env, "rect_corner_margin_max", 0.0)),
            "rect_corner_proximity_distance": float(getattr(env, "rect_corner_proximity_distance", 0.4)),
            "rect_corner_speed_min": float(getattr(env, "rect_corner_speed_min", 0.05)),
            "rect_corner_alignment_power": float(getattr(env, "rect_corner_alignment_power", 1.0)),
            "rect_dual_edge_cbf_enabled": bool(getattr(env, "rect_dual_edge_cbf_enabled", False)),
            "rect_dual_edge_proximity_distance": float(getattr(env, "rect_dual_edge_proximity_distance", 0.0)),
            "rect_smooth_tau": float(getattr(env, "rect_smooth_tau", 0.1)),
        }
        ctx.update(dict(self.config.skill_context))
        return ctx

    def _current_runtime_ctx(self) -> Dict[str, Any]:
        ctx = self._default_skill_context()
        ctx["wind_accel"] = self.core_env.env.get_current_wind_accel()
        return ctx

    def _build_default_action_adapter(self):
        env = self.core_env.env
        lidar_cfg = {
            "use_fitted_geometry": bool(getattr(env, "lidar_cbf_use_fitted_geometry", False)),
            "point_radius": float(getattr(env, "lidar_cbf_point_radius", 0.0)),
            "top_k": int(getattr(env, "lidar_cbf_top_k", 0)),
            "min_segment_points": int(getattr(env, "lidar_cbf_min_segment_points", 2)),
            "line_fit_max_residual": float(getattr(env, "lidar_cbf_line_fit_max_residual", 0.08)),
            "circle_fit_max_residual": float(getattr(env, "lidar_cbf_circle_fit_max_residual", 0.08)),
            "circle_radius_min": float(getattr(env, "lidar_cbf_circle_radius_min", 0.05)),
            "circle_radius_max": float(getattr(env, "lidar_cbf_circle_radius_max", 100.0)),
        }
        constraint_builder = ConstraintBuilder(
            d_min_agent=float(self.config.d_min_agent),
            d_safe_obs=float(self.config.d_safe_obs),
            u_min=np.asarray([-env.action_limit, -env.action_limit], dtype=np.float32),
            u_max=np.asarray([env.action_limit, env.action_limit], dtype=np.float32),
            lidar_cbf_config=lidar_cfg,
        )
        diff_qp_solver = TorchDifferentiableQPSolver(**dict(self.config.solver_config))
        neighbor_radius = (
            float(self.config.neighbor_perception_radius)
            if self.config.neighbor_perception_radius is not None
            else float(getattr(env, "neighbor_radius", 0.0))
        )
        obstacle_range = (
            float(self.config.obstacle_perception_range)
            if self.config.obstacle_perception_range is not None
            else float(env.lidar.max_range)
        )
        decoder_kwargs = dict(self.config.decoder_config)
        decoder_kwargs.setdefault("action_dim", 2)
        return TorchDiffQPActionAdapter(
            decoder=LowQPDecoder(**decoder_kwargs),
            diff_qp_solver=diff_qp_solver,
            constraint_builder=constraint_builder,
            neighbor_perception_radius=neighbor_radius,
            obstacle_perception_range=obstacle_range,
            torch_device=str(self.config.torch_device),
        )

    def _action_adapter_name(self) -> str:
        return type(self.action_adapter).__name__

    def _available_skill_ids(
        self,
        agent_id: int,
        state: AgentState,
        runtime_ctx: Mapping[str, Any] | None = None,
    ) -> List[int]:
        ctx = dict(runtime_ctx or self._current_runtime_ctx())
        available: List[int] = []
        for skill_id, skill in sorted(self.skill_runtime.skill_by_id.items()):
            if bool(skill.initiation_set_fn(state, dict(ctx))):
                available.append(int(skill_id))
        return available

    def current_skill_ids(self) -> Dict[int, int]:
        return {int(agent_id): int(skill_id) for agent_id, skill_id in self._skill_ids.items()}

    def get_available_skill_ids(self, agent_id: int) -> List[int]:
        agent_id = int(agent_id)
        states = self.core_env.get_states()
        if agent_id not in states:
            raise KeyError(f"unknown agent_id {agent_id}")
        return self._available_skill_ids(agent_id, states[agent_id], self._current_runtime_ctx())

    def current_actor_obs(self) -> Dict[int, np.ndarray]:
        return self.core_env.build_low_actor_obs(skill_ids=self._skill_ids, n_skills=self.config.n_skills)

    def current_infos(self) -> Dict[int, Dict[str, Any]]:
        critic_state = self.state()
        switch_mask = self.get_switch_mask()
        return {
            int(agent_id): {
                "critic_state": critic_state.copy(),
                "skill_id": int(self._skill_ids.get(agent_id, self.config.default_skill_id)),
                "switch_required": bool(switch_mask[int(agent_id)]),
                "pending_switch_agents": np.asarray(self.pending_switch_agents, dtype=np.int64),
                "action_adapter": self._action_adapter_name(),
            }
            for agent_id in self.possible_agents
        }

    @property
    def pending_switch_agents(self) -> List[int]:
        return sorted(int(agent_id) for agent_id in self._pending_reactivate if agent_id not in self._done_agents)

    def has_pending_switch(self) -> bool:
        return len(self.pending_switch_agents) > 0

    def get_switch_mask(self) -> Dict[int, bool]:
        pending = set(self.pending_switch_agents)
        return {int(agent_id): bool(agent_id in pending) for agent_id in self.possible_agents}

    def set_skill_map(
        self,
        skill_ids: Mapping[int, int],
        *,
        validate: bool = True,
        require_all: bool = False,
    ) -> None:
        states = self.core_env.get_states()
        runtime_ctx = self._current_runtime_ctx()
        if require_all:
            target_agents = [agent_id for agent_id in self.possible_agents if agent_id not in self._done_agents]
        else:
            pending = self.pending_switch_agents
            target_agents = pending if pending else [agent_id for agent_id in self.possible_agents if agent_id not in self._done_agents]

        for agent_id in target_agents:
            if agent_id not in skill_ids:
                raise KeyError(f"missing skill_id for agent {agent_id}")
            skill_id = int(skill_ids[agent_id])
            if validate:
                available = self._available_skill_ids(agent_id, states[agent_id], runtime_ctx)
                if skill_id not in available:
                    raise ValueError(
                        f"agent {agent_id} selected unavailable skill {skill_id}; available={available}"
                    )
            self.skill_runtime.activate_skill(
                agent_id=agent_id,
                skill_id=skill_id,
                state=states[agent_id],
                extra_ctx=runtime_ctx,
            )
            self._skill_ids[agent_id] = skill_id
            self._pending_reactivate.discard(agent_id)

    def _initialize_skills(self, skill_ids: Mapping[int, int] | None = None) -> None:
        states = self.core_env.get_states()
        runtime_ctx = self._current_runtime_ctx()
        chosen: Dict[int, int] = {}
        for agent_id in self.possible_agents:
            if agent_id in self._done_agents:
                continue
            if skill_ids is not None and agent_id in skill_ids:
                chosen_skill = int(skill_ids[agent_id])
            else:
                available = self._available_skill_ids(agent_id, states[agent_id], runtime_ctx)
                if not available:
                    raise RuntimeError(f"agent {agent_id} has no initiable skill at reset")
                preferred = int(self.config.default_skill_id)
                chosen_skill = preferred if preferred in available else int(available[0])
            chosen[agent_id] = chosen_skill
        self.set_skill_map(chosen, validate=True, require_all=True)

    def _ensure_reactivated_skills(self) -> None:
        if not self._pending_reactivate:
            return
        if not self.config.reactivate_same_skill_on_switch:
            raise RuntimeError(
                f"agents {sorted(self._pending_reactivate)} require a skill switch; "
                "set_skill_map(...) must be called before the next low-level step"
            )
        states = self.core_env.get_states()
        runtime_ctx = self._current_runtime_ctx()
        for agent_id in sorted(self._pending_reactivate):
            if agent_id in self._done_agents or agent_id in self._frozen_agents:
                continue
            skill_id = int(self._skill_ids[agent_id])
            available = self._available_skill_ids(agent_id, states[agent_id], runtime_ctx)
            if skill_id not in available:
                raise RuntimeError(
                    f"agent {agent_id} cannot reactivate skill {skill_id}; available={available}"
                )
            self.skill_runtime.activate_skill(
                agent_id=agent_id,
                skill_id=skill_id,
                state=states[agent_id],
                extra_ctx=runtime_ctx,
            )
        self._pending_reactivate.clear()

    def reset(self, seed: int | None = None, options: Mapping[str, Any] | None = None):
        states = None if options is None else options.get("states")
        obstacles = None if options is None else options.get("obstacles")
        skill_ids = None if options is None else options.get("skill_ids")
        obs, _ = self.core_env.reset_scene(seed=seed, states=states, obstacles=obstacles, options=options)
        del obs

        self.skill_runtime.reset(self.possible_agents)
        self.agents = list(self.possible_agents)
        self._done_agents = set()
        self._frozen_agents = set()
        self._pending_reactivate = set()
        self._last_skill_outputs = {}
        self._last_adapter_info = {}
        self._skill_ids = {}
        self._initialize_skills(skill_ids=skill_ids)

        return self.current_actor_obs(), self.current_infos()

    def state(self) -> np.ndarray:
        return self.core_env.build_low_critic_obs(
            skill_ids=self._skill_ids,
            n_skills=self.config.n_skills,
            include_local_low_obs=bool(self.config.include_local_obs_in_critic),
        )

    def step(self, actions: Mapping[int, np.ndarray] | np.ndarray):
        self._ensure_reactivated_skills()

        if isinstance(actions, np.ndarray):
            arr = np.asarray(actions, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(self.agent_num, -1)
            phi_actions = {int(agent_id): arr[idx].reshape(-1) for idx, agent_id in enumerate(self.possible_agents)}
        else:
            phi_actions = {
                int(agent_id): np.asarray(action, dtype=np.float32).reshape(-1)
                for agent_id, action in actions.items()
            }

        for agent_id in self.possible_agents:
            if agent_id in self._done_agents:
                continue
            if agent_id not in phi_actions:
                raise KeyError(f"missing low-level action for agent {agent_id}")

        states = self.core_env.get_states()
        obs_low = self.core_env.get_obs_low()
        runtime_ctx = self._current_runtime_ctx()
        active_agent_ids = [aid for aid in self.possible_agents if aid not in self._done_agents and aid not in self._frozen_agents]

        skill_targets = self.skill_runtime.control_targets(
            states={aid: states[aid] for aid in active_agent_ids},
            obs_low={aid: obs_low[aid] for aid in active_agent_ids},
            runtime_ctx=runtime_ctx,
        ) if active_agent_ids else {}

        executed_actions = {int(agent_id): np.zeros(2, dtype=np.float32) for agent_id in self.possible_agents}
        adapter_info: Dict[int, Dict[str, Any]] = {int(agent_id): {} for agent_id in self.possible_agents}
        if active_agent_ids:
            adapted_actions, adapted_info = self.action_adapter(
                {aid: states[aid] for aid in active_agent_ids},
                {aid: obs_low[aid] for aid in active_agent_ids},
                skill_targets,
                {aid: phi_actions[aid] for aid in active_agent_ids},
                self.core_env,
            )
            for agent_id in active_agent_ids:
                executed_actions[int(agent_id)] = np.asarray(adapted_actions[int(agent_id)], dtype=np.float32).reshape(2)
                adapter_info[int(agent_id)] = dict(adapted_info.get(int(agent_id), {}))

        step_res = self.core_env.step_low_level(executed_actions)
        next_states = self.core_env.get_states()
        next_obs_low = self.core_env.get_obs_low()

        skill_outputs = self.skill_runtime.step_all(
            states={aid: next_states[aid] for aid in active_agent_ids},
            obs_low={aid: next_obs_low[aid] for aid in active_agent_ids},
            executed_actions={aid: executed_actions[aid] for aid in active_agent_ids},
            runtime_ctx=runtime_ctx,
        ) if active_agent_ids else {}

        rewards = {int(agent_id): 0.0 for agent_id in self.possible_agents}
        terminations = {int(agent_id): False for agent_id in self.possible_agents}
        truncations = {int(agent_id): bool(step_res.truncated) for agent_id in self.possible_agents}
        infos: Dict[int, Dict[str, Any]] = {int(agent_id): {} for agent_id in self.possible_agents}

        newly_frozen = {
            aid for aid in active_agent_ids
            if bool(step_res.info.get("reach_flags", {}).get(aid, False))
        }
        if newly_frozen and self.config.freeze_on_reach and hasattr(self.core_env.env, "freeze_agents"):
            self.core_env.env.freeze_agents(sorted(newly_frozen))
        self._frozen_agents.update(newly_frozen)

        for agent_id in active_agent_ids:
            skill_out = skill_outputs[agent_id]
            ext_reward = float(step_res.rewards.get(agent_id, 0.0))
            int_reward = float(skill_out.intrinsic_reward)
            total_reward = ext_reward + float(self.config.intrinsic_reward_coef) * int_reward
            rewards[agent_id] = total_reward

            unsafe = bool(step_res.info.get("unsafe_flags", {}).get(agent_id, False))
            reached = bool(step_res.info.get("reach_flags", {}).get(agent_id, False))
            terminated_by_skill = bool(skill_out.beta)
            switch_required = bool(terminated_by_skill)
            if switch_required:
                self._pending_reactivate.add(agent_id)

            terminations[agent_id] = bool(unsafe or reached)
            if terminations[agent_id]:
                self._done_agents.add(agent_id)
                self._pending_reactivate.discard(agent_id)

            infos[agent_id] = {
                "critic_state": self.state().copy(),
                "skill_id": int(self._skill_ids[agent_id]),
                "switch_required": switch_required,
                "terminated_by_skill": terminated_by_skill,
                "intrinsic_reward": int_reward,
                "reward_ext": ext_reward,
                "reward_total": total_reward,
                "tau": int(skill_out.tau),
                "u_ref_skill": np.asarray(skill_out.u_ref_skill, dtype=np.float32).reshape(2).copy(),
                "safety_constraints": dict(skill_out.safety_constraints),
                "phi": np.asarray(phi_actions[agent_id], dtype=np.float32).copy(),
                "executed_action": np.asarray(executed_actions[agent_id], dtype=np.float32).reshape(2).copy(),
                **adapter_info.get(agent_id, {}),
            }

        for agent_id in self.possible_agents:
            infos.setdefault(agent_id, {})
            infos[agent_id].setdefault("critic_state", self.state().copy())
            infos[agent_id].setdefault("skill_id", int(self._skill_ids.get(agent_id, self.config.default_skill_id)))
            infos[agent_id].setdefault("switch_required", bool(agent_id in self._pending_reactivate))
            infos[agent_id].setdefault("pending_switch_agents", np.asarray(self.pending_switch_agents, dtype=np.int64))
            infos[agent_id].setdefault("phi", np.asarray(phi_actions.get(agent_id, np.zeros(self.phi_dim, dtype=np.float32)), dtype=np.float32))
            infos[agent_id].setdefault("executed_action", np.asarray(executed_actions.get(agent_id, np.zeros(2, dtype=np.float32)), dtype=np.float32))

        self.agents = [aid for aid in self.possible_agents if aid not in self._done_agents]
        self._last_skill_outputs = {
            int(agent_id): {
                "skill_id": int(skill_outputs[agent_id].skill_id),
                "tau": int(skill_outputs[agent_id].tau),
                "beta": bool(skill_outputs[agent_id].beta),
                "intrinsic_reward": float(skill_outputs[agent_id].intrinsic_reward),
            }
            for agent_id in skill_outputs
        }
        self._last_adapter_info = {int(agent_id): dict(info) for agent_id, info in adapter_info.items()}

        next_actor_obs = self.core_env.build_low_actor_obs(skill_ids=self._skill_ids, n_skills=self.config.n_skills)
        return next_actor_obs, rewards, terminations, truncations, infos
