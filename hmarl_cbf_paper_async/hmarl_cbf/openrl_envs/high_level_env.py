from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

try:
    from pettingzoo import ParallelEnv
except ImportError:  # pragma: no cover - optional dependency
    ParallelEnv = object  # type: ignore[assignment,misc]

from hmarl_cbf.env.obstacles import normalize_obstacle, obstacle_corners
from hmarl_cbf.openrl_envs.core_env import CoreEnv
from hmarl_cbf.skills import SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.control.sync_coordinator import SyncCoordinator
from hmarl_cbf.types import AgentObsLow, AgentState, SkillSpec


LowLevelExecutor = Callable[
    [Mapping[int, AgentState], Mapping[int, AgentObsLow], Mapping[int, Dict[str, Any]], CoreEnv],
    Dict[int, np.ndarray],
]


@dataclass(slots=True)
class HighLevelOpenRLEnvConfig:
    n_skills: int
    coordinator_mode: str = "sync"
    t_sync_max: int = 10
    gamma_high: float = 0.99
    max_low_steps_per_high_step: int = 0
    freeze_on_reach: bool = True
    d_safe_obs: float = 0.6
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
    skill_context: Dict[str, Any] = field(default_factory=dict)


class HighLevelOpenRLEnv(ParallelEnv):  # type: ignore[misc]
    """OpenRL-friendly high-level multi-agent environment view.

    One high-level step:
    - receives skill selections
    - activates newly switching options
    - rolls the underlying physics forward for multiple low-level steps
    - returns option-level rewards and next high-level observations
    """

    metadata = {"name": "hmarl_cbf_high_level_openrl"}

    def __init__(
        self,
        core_env: CoreEnv,
        config: HighLevelOpenRLEnvConfig,
        *,
        skill_library: Iterable[SkillSpec] | None = None,
        low_level_executor: LowLevelExecutor | None = None,
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
        self.coordinator = SyncCoordinator(
            num_agents=int(self.core_env.env.n_agents),
            t_sync_max=int(self.config.t_sync_max),
            mode=str(self.config.coordinator_mode),
        )
        if low_level_executor is None:
            raise ValueError("HighLevelOpenRLEnv requires an explicit low_level_executor")
        self.low_level_executor = low_level_executor

        actor_dim = 4 + 2 + (int(self.core_env.env.obs_builder.max_neighbors) * 4)
        critic_dim = (
            int(self.core_env.env.n_agents) * CoreEnv._AGENT_STATE_DIM
            + int(self.core_env.obstacle_slots) * CoreEnv._OBSTACLE_DIM
            + 2
            + 2
        )
        self._actor_obs_dim = int(actor_dim)
        self._critic_obs_dim = int(critic_dim)

        if spaces is not None:
            self.observation_spaces = {
                agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=(self._actor_obs_dim,), dtype=np.float32)
                for agent_id in self.possible_agents
            }
            self.action_spaces = {
                agent_id: spaces.Discrete(int(self.config.n_skills))
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

        self._pending_switch_agents: set[int] = set(self.possible_agents)
        self._done_agents: set[int] = set()
        self._frozen_agents: set[int] = set()
        self._option_start_goal_dist: Dict[int, float] = {}
        self._option_start_boundary_clearance: Dict[int, float] = {}
        self._option_start_blocked_score: Dict[int, float] = {}
        self._round_return_ext: Dict[int, float] = {}
        self._round_discount: Dict[int, float] = {}
        self._option_speed_sum: Dict[int, float] = {}
        self._option_speed_count: Dict[int, int] = {}
        self._last_low_step_records: List[Dict[str, Any]] = []

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
            "d_safe_obs": float(self.config.d_safe_obs),
            "obstacle_perception_range": float(env.lidar.max_range),
            "lidar_obstacle_cbf_enabled": bool(getattr(env, "lidar_obstacle_cbf_enabled", True)),
            "lidar_cbf_point_radius": float(getattr(env, "lidar_cbf_point_radius", 0.0)),
            "lidar_cbf_top_k": int(getattr(env, "lidar_cbf_top_k", 3)),
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

    @staticmethod
    def _boundary_clearance(state: AgentState, world_size: float) -> float:
        pos = np.asarray(state.position, dtype=np.float32).reshape(2)
        return float(world_size - max(abs(float(pos[0])), abs(float(pos[1]))))

    def _compute_blocked_score(
        self,
        state: AgentState,
        obstacles: List[Dict[str, Any]],
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
        lookahead = float(max(1e-3, self.config.high_trap_blocked_lookahead))
        lateral_window = float(max(1e-3, self.config.high_trap_blocked_lateral_window))
        extra_margin = float(max(0.0, self.config.high_trap_blocked_extra_margin))
        required_width = 2.0 * float(state.radius + self.config.d_safe_obs + extra_margin)

        intervals: List[tuple[float, float]] = []
        for obstacle in obstacles:
            obs = normalize_obstacle(dict(obstacle))
            inflated = float(state.radius + self.config.d_safe_obs + extra_margin)
            if str(obs["type"]) in ("circle", "point", "lidar_point", "lidar_circle"):
                center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
                radius = float(obs["radius"])
                rel = center - pos
                longitudinal = float(np.dot(rel, e_goal))
                if longitudinal <= 0.0 or longitudinal > lookahead:
                    continue
                lateral = float(np.dot(rel, e_perp))
                left = max(-lateral_window, lateral - (radius + inflated))
                right = min(lateral_window, lateral + (radius + inflated))
            elif str(obs["type"]) == "lidar_line":
                start = np.asarray(obs["start"], dtype=np.float32).reshape(2)
                end = np.asarray(obs["end"], dtype=np.float32).reshape(2)
                rel_points = np.stack([start - pos, end - pos], axis=0)
                longitudinal_vals = rel_points @ e_goal.reshape(2, 1)
                lateral_vals = rel_points @ e_perp.reshape(2, 1)
                longitudinal_min = float(np.min(longitudinal_vals)) - inflated
                longitudinal_max = float(np.max(longitudinal_vals)) + inflated
                if longitudinal_max <= 0.0 or longitudinal_min > lookahead:
                    continue
                left = max(-lateral_window, float(np.min(lateral_vals)) - inflated)
                right = min(lateral_window, float(np.max(lateral_vals)) + inflated)
            else:
                corners = obstacle_corners(obs)
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

    def _reset_option_trackers(self, agent_ids: Iterable[int], states: Mapping[int, AgentState]) -> None:
        world_size = float(self.core_env.env.world_size)
        obstacles = self.core_env.get_obstacles()
        for agent_id in agent_ids:
            state = states[int(agent_id)]
            self._option_start_goal_dist[int(agent_id)] = float(np.linalg.norm(state.goal - state.position))
            self._option_start_boundary_clearance[int(agent_id)] = self._boundary_clearance(state, world_size)
            self._option_start_blocked_score[int(agent_id)] = self._compute_blocked_score(state, obstacles)
            self._round_return_ext[int(agent_id)] = 0.0
            self._round_discount[int(agent_id)] = 1.0
            self._option_speed_sum[int(agent_id)] = 0.0
            self._option_speed_count[int(agent_id)] = 0

    def _current_runtime_ctx(self) -> Dict[str, Any]:
        ctx = self._default_skill_context()
        ctx["wind_accel"] = self.core_env.env.get_current_wind_accel()
        return ctx

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

    def get_action_mask(self) -> Dict[int, np.ndarray]:
        states = self.core_env.get_states()
        runtime_ctx = self._current_runtime_ctx()
        masks: Dict[int, np.ndarray] = {}
        for agent_id in self.possible_agents:
            mask = np.zeros(int(self.config.n_skills), dtype=np.float32)
            if agent_id not in self._done_agents and agent_id in self._pending_switch_agents:
                for skill_id in self._available_skill_ids(agent_id, states[agent_id], runtime_ctx):
                    if 0 <= int(skill_id) < int(self.config.n_skills):
                        mask[int(skill_id)] = 1.0
            masks[int(agent_id)] = mask
        return masks

    def valid_skill_ids(self) -> Dict[int, List[int]]:
        states = self.core_env.get_states()
        runtime_ctx = self._current_runtime_ctx()
        valid: Dict[int, List[int]] = {}
        for agent_id in self.possible_agents:
            if agent_id not in self._done_agents and agent_id in self._pending_switch_agents:
                valid[int(agent_id)] = self._available_skill_ids(agent_id, states[agent_id], runtime_ctx)
            else:
                valid[int(agent_id)] = []
        return valid

    def current_skill_ids(self) -> Dict[int, int]:
        skill_ids: Dict[int, int] = {}
        for agent_id in self.possible_agents:
            if agent_id in self._done_agents:
                continue
            try:
                skill_ids[int(agent_id)] = int(self.skill_runtime.current_skill_id(int(agent_id)))
            except KeyError:
                continue
        return skill_ids

    def decision_agent_ids(self) -> List[int]:
        if len(self._pending_switch_agents) == 0:
            self._pending_switch_agents = {aid for aid in self.possible_agents if aid not in self._done_agents}
        return [
            int(aid)
            for aid in self.possible_agents
            if aid in self._pending_switch_agents and aid not in self._done_agents
        ]

    def _activate_skills(self, chosen_skills: Mapping[int, int]) -> None:
        states = self.core_env.get_states()
        runtime_ctx = self._current_runtime_ctx()
        activatable = [aid for aid in self._pending_switch_agents if aid not in self._done_agents]
        for agent_id in activatable:
            chosen = int(chosen_skills[int(agent_id)])
            available = self._available_skill_ids(int(agent_id), states[int(agent_id)], runtime_ctx)
            if chosen not in available:
                raise ValueError(
                    f"agent {agent_id} selected unavailable skill {chosen}; available={available}"
                )
            self.skill_runtime.activate_skill(
                agent_id=int(agent_id),
                skill_id=chosen,
                state=states[int(agent_id)],
                extra_ctx=runtime_ctx,
            )
        self._reset_option_trackers(activatable, states)

    def reset(self, seed: int | None = None, options: Mapping[str, Any] | None = None):
        states = None if options is None else options.get("states")
        obstacles = None if options is None else options.get("obstacles")
        obs, _ = self.core_env.reset_scene(seed=seed, states=states, obstacles=obstacles, options=options)
        del obs
        self.skill_runtime.reset(self.possible_agents)
        self.coordinator.reset()
        self.agents = list(self.possible_agents)
        self._pending_switch_agents = set(self.possible_agents)
        self._done_agents = set()
        self._frozen_agents = set()
        self._option_start_goal_dist.clear()
        self._option_start_boundary_clearance.clear()
        self._option_start_blocked_score.clear()
        self._round_return_ext.clear()
        self._round_discount.clear()
        self._option_speed_sum.clear()
        self._option_speed_count.clear()
        self._last_low_step_records = []

        obs_high = self.core_env.build_high_actor_obs()
        action_masks = self.get_action_mask()
        valid_skills = self.valid_skill_ids()
        infos = {
            int(agent_id): {
                "critic_state": self.state().copy(),
                "switch_required": True,
                "switch_required_next": True,
                "option_k": int(self.coordinator.option_k[int(agent_id)]),
                "action_mask": action_masks[int(agent_id)].copy(),
                "action_masks": action_masks[int(agent_id)].copy(),
                "valid_skill_ids": list(valid_skills[int(agent_id)]),
            }
            for agent_id in self.possible_agents
        }
        return obs_high, infos

    def state(self) -> np.ndarray:
        return self.core_env.build_high_critic_obs()

    @property
    def last_low_step_records(self) -> List[Dict[str, Any]]:
        return list(self._last_low_step_records)

    def step(self, actions: Mapping[int, int]):
        required_agents = self.decision_agent_ids()
        missing = [aid for aid in required_agents if int(aid) not in actions]
        if missing:
            raise KeyError(f"missing high-level actions for agents requiring a switch: {missing}")
        chosen_skills = {int(agent_id): int(actions[int(agent_id)]) for agent_id in required_agents}
        self._activate_skills(chosen_skills)

        rewards = {int(agent_id): 0.0 for agent_id in self.possible_agents}
        terminations = {int(agent_id): False for agent_id in self.possible_agents}
        truncations = {int(agent_id): False for agent_id in self.possible_agents}
        info_by_agent: Dict[int, Dict[str, Any]] = {int(agent_id): {} for agent_id in self.possible_agents}

        max_steps = int(self.config.max_low_steps_per_high_step)
        if max_steps <= 0:
            max_steps = int(self.core_env.env.horizon)

        switched_agents: set[int] = set()
        newly_frozen: set[int] = set()
        forced_end = False
        last_sync = None
        low_step_records: List[Dict[str, Any]] = []

        for _ in range(max_steps):
            states = self.core_env.get_states()
            obs_low = self.core_env.get_obs_low()
            active_agent_ids = [aid for aid in self.possible_agents if aid not in self._done_agents and aid not in self._frozen_agents]
            option_k_before_step = {int(aid): int(self.coordinator.option_k[int(aid)]) for aid in self.possible_agents}
            runtime_ctx = self._current_runtime_ctx()
            skill_targets = self.skill_runtime.control_targets(
                states={aid: states[aid] for aid in active_agent_ids},
                obs_low={aid: obs_low[aid] for aid in active_agent_ids},
                runtime_ctx=runtime_ctx,
            ) if active_agent_ids else {}

            low_actions = {int(agent_id): np.zeros(2, dtype=np.float32) for agent_id in self.possible_agents}
            adapter_info: Dict[int, Dict[str, Any]] = {int(agent_id): {} for agent_id in self.possible_agents}
            if active_agent_ids:
                if hasattr(self.low_level_executor, "prepare_step_context"):
                    self.low_level_executor.prepare_step_context(skill_ids=self.current_skill_ids())
                executed_result = self.low_level_executor(
                    {aid: states[aid] for aid in active_agent_ids},
                    {aid: obs_low[aid] for aid in active_agent_ids},
                    skill_targets,
                    self.core_env,
                )
                if isinstance(executed_result, tuple) and len(executed_result) == 2:
                    executed, returned_info = executed_result
                    for agent_id, item in dict(returned_info).items():
                        adapter_info[int(agent_id)] = dict(item)
                else:
                    executed = executed_result
                for agent_id, action in executed.items():
                    low_actions[int(agent_id)] = np.asarray(action, dtype=np.float32).reshape(2)

            step_res = self.core_env.step_low_level(low_actions)
            next_states = self.core_env.get_states()
            next_obs_low = self.core_env.get_obs_low()

            skill_out = self.skill_runtime.step_all(
                states={aid: next_states[aid] for aid in active_agent_ids},
                obs_low={aid: next_obs_low[aid] for aid in active_agent_ids},
                executed_actions={aid: low_actions[aid] for aid in active_agent_ids},
                runtime_ctx=runtime_ctx,
            ) if active_agent_ids else {}

            if active_agent_ids and hasattr(self.low_level_executor, "record_transition"):
                self.low_level_executor.record_transition(
                    states={aid: states[aid] for aid in active_agent_ids},
                    obs_low={aid: obs_low[aid] for aid in active_agent_ids},
                    skill_targets={aid: dict(skill_targets[aid]) for aid in active_agent_ids},
                    executed_actions={aid: np.asarray(low_actions[aid], dtype=np.float32).reshape(2) for aid in active_agent_ids},
                    step_res=step_res,
                    next_states={aid: next_states[aid] for aid in active_agent_ids},
                    next_obs_low={aid: next_obs_low[aid] for aid in active_agent_ids},
                    skill_outputs={aid: skill_out[aid] for aid in active_agent_ids},
                    adapter_info={aid: dict(adapter_info.get(aid, {})) for aid in active_agent_ids},
                    option_k_by_agent={aid: int(option_k_before_step.get(aid, 0)) for aid in active_agent_ids},
                    done_agents=set(self._done_agents),
                )

            low_step_records.append(
                {
                    "positions": {
                        int(aid): np.asarray(next_states[aid].position, dtype=np.float32).reshape(2).copy()
                        for aid in self.possible_agents
                    },
                    "velocities": {
                        int(aid): np.asarray(next_states[aid].velocity, dtype=np.float32).reshape(2).copy()
                        for aid in self.possible_agents
                    },
                    "rewards": {int(aid): float(step_res.rewards.get(aid, 0.0)) for aid in self.possible_agents},
                    "unsafe_flags": {
                        int(aid): bool(step_res.info.get("unsafe_flags", {}).get(aid, False))
                        for aid in self.possible_agents
                    },
                    "reach_flags": {
                        int(aid): bool(step_res.info.get("reach_flags", {}).get(aid, False))
                        for aid in self.possible_agents
                    },
                    "safety_metrics": {
                        int(aid): {
                            "min_h_agent": float(
                                step_res.info.get("safety_metrics", {}).get(aid).min_h_agent
                                if step_res.info.get("safety_metrics", {}).get(aid) is not None
                                else 0.0
                            ),
                            "min_h_obstacle": float(
                                step_res.info.get("safety_metrics", {}).get(aid).min_h_obstacle
                                if step_res.info.get("safety_metrics", {}).get(aid) is not None
                                else 0.0
                            ),
                        }
                        for aid in self.possible_agents
                    },
                    "adapter_info": {int(aid): dict(adapter_info.get(aid, {})) for aid in self.possible_agents},
                    "step_count": int(getattr(self.core_env.env, "step_count", 0)),
                    "truncated": bool(step_res.truncated),
                    "terminated": bool(step_res.terminated),
                    "wind_accel": np.asarray(
                        step_res.info.get("wind_accel", self.core_env.env.get_current_wind_accel()),
                        dtype=np.float32,
                    ).reshape(2).copy(),
                }
            )

            beta = {aid: (bool(skill_out[aid].beta) if aid in skill_out else False) for aid in self.possible_agents}
            last_sync = self.coordinator.step(beta)
            switched_agents = set(int(aid) for aid in last_sync.switch_agents)

            for agent_id in active_agent_ids:
                self._round_return_ext[agent_id] += self._round_discount[agent_id] * float(step_res.rewards[agent_id])
                self._round_discount[agent_id] *= float(self.config.gamma_high)
                self._option_speed_sum[agent_id] += float(np.linalg.norm(next_states[agent_id].velocity))
                self._option_speed_count[agent_id] += 1

            newly_frozen = {
                aid for aid in active_agent_ids
                if bool(step_res.info.get("reach_flags", {}).get(aid, False))
            }
            if newly_frozen and self.config.freeze_on_reach and hasattr(self.core_env.env, "freeze_agents"):
                self.core_env.env.freeze_agents(sorted(newly_frozen))
            self._frozen_agents.update(newly_frozen)

            forced_end = bool(step_res.terminated or step_res.truncated)
            if forced_end or switched_agents or newly_frozen:
                break

        close_agents = set(switched_agents) | set(newly_frozen)
        if forced_end:
            close_agents |= {aid for aid in self.possible_agents if aid not in self._done_agents}

        final_states = self.core_env.get_states()
        final_obstacles = self.core_env.get_obstacles()
        world_size = float(self.core_env.env.world_size)

        for agent_id in close_agents:
            if agent_id in self._done_agents:
                continue
            goal_dist_end = float(np.linalg.norm(final_states[agent_id].goal - final_states[agent_id].position))
            goal_dist_start = float(self._option_start_goal_dist.get(agent_id, goal_dist_end))
            option_progress_bonus = float(self.config.high_option_progress_coef) * float(goal_dist_start - goal_dist_end)

            boundary_clearance_end = self._boundary_clearance(final_states[agent_id], world_size)
            boundary_clearance_start = float(self._option_start_boundary_clearance.get(agent_id, boundary_clearance_end))
            blocked_score_end = self._compute_blocked_score(final_states[agent_id], final_obstacles)
            blocked_score_start = float(self._option_start_blocked_score.get(agent_id, blocked_score_end))

            boundary_recovery_bonus = 0.0
            threshold = float(self.config.high_option_boundary_threshold)
            if threshold > 0.0 and boundary_clearance_start <= threshold:
                boundary_recovery_bonus = float(self.config.high_option_boundary_recovery_coef) * float(
                    boundary_clearance_end - boundary_clearance_start
                )

            trap_relief_bonus = float(self.config.high_option_trap_relief_coef) * float(blocked_score_start - blocked_score_end)
            trap_enter_penalty = float(self.config.high_option_trap_enter_coef) * float(
                blocked_score_start * max(0.0, goal_dist_start - goal_dist_end)
            )

            option_progress = float(goal_dist_start - goal_dist_end)
            avg_speed = float(self._option_speed_sum.get(agent_id, 0.0) / max(1, self._option_speed_count.get(agent_id, 0)))
            stuck_penalty = 0.0
            if (
                float(self.config.high_option_stuck_penalty_coef) > 0.0
                and blocked_score_end >= float(self.config.high_option_stuck_blocked_threshold)
                and option_progress <= float(self.config.high_option_stuck_progress_threshold)
                and avg_speed <= float(self.config.high_option_stuck_speed_threshold)
            ):
                stuck_penalty = float(self.config.high_option_stuck_penalty_coef)

            total_reward = (
                float(self._round_return_ext.get(agent_id, 0.0))
                + option_progress_bonus
                + boundary_recovery_bonus
                + trap_relief_bonus
                - trap_enter_penalty
                - stuck_penalty
            )
            rewards[agent_id] = float(total_reward)
            if forced_end:
                terminations[agent_id] = bool(self.core_env.last_terminated)
                truncations[agent_id] = bool(self.core_env.last_truncated)
            else:
                terminations[agent_id] = bool(self.core_env.last_info.get("unsafe_flags", {}).get(agent_id, False))
                truncations[agent_id] = bool(self.core_env.last_truncated)
            if bool(self.core_env.last_info.get("reach_flags", {}).get(agent_id, False)):
                terminations[agent_id] = True

            info_by_agent[agent_id] = {
                "chosen_skill_id": int(chosen_skills.get(agent_id, 0)),
                "actual_skill_id": int(chosen_skills.get(agent_id, 0)),
                "option_k": int(self.coordinator.option_k[int(agent_id)]),
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
                "switch_required_next": bool(agent_id in switched_agents),
                "critic_state": self.state().copy(),
            }

            if terminations[agent_id] or truncations[agent_id]:
                self._done_agents.add(agent_id)

        for agent_id in self.possible_agents:
            info_by_agent.setdefault(agent_id, {})
            info_by_agent[agent_id].setdefault("critic_state", self.state().copy())
            info_by_agent[agent_id].setdefault("switch_required_next", bool(agent_id in switched_agents))
            info_by_agent[agent_id].setdefault("chosen_skill_id", int(chosen_skills.get(agent_id, 0)))
            info_by_agent[agent_id].setdefault("actual_skill_id", int(chosen_skills.get(agent_id, 0)))
            info_by_agent[agent_id].setdefault("option_k", int(self.coordinator.option_k[int(agent_id)]))
            if forced_end:
                terminations[agent_id] = bool(self.core_env.last_terminated)
                truncations[agent_id] = bool(self.core_env.last_truncated)

        self._pending_switch_agents = {aid for aid in switched_agents if aid not in self._done_agents}
        if not self._pending_switch_agents and not forced_end:
            self._pending_switch_agents = {aid for aid in self.possible_agents if aid not in self._done_agents}
        self._last_low_step_records = low_step_records

        if forced_end:
            self._done_agents = set(self.possible_agents)

        self.agents = [aid for aid in self.possible_agents if aid not in self._done_agents]
        next_obs_high = self.core_env.build_high_actor_obs()
        next_action_masks = self.get_action_mask()
        next_valid_skills = self.valid_skill_ids()
        for agent_id in self.possible_agents:
            info_by_agent[agent_id]["switch_required"] = bool(
                agent_id in self._pending_switch_agents and agent_id not in self._done_agents
            )
            info_by_agent[agent_id]["option_k"] = int(self.coordinator.option_k[int(agent_id)])
            info_by_agent[agent_id]["action_mask"] = next_action_masks[int(agent_id)].copy()
            info_by_agent[agent_id]["action_masks"] = next_action_masks[int(agent_id)].copy()
            info_by_agent[agent_id]["valid_skill_ids"] = list(next_valid_skills[int(agent_id)])
        return next_obs_high, rewards, terminations, truncations, info_by_agent
