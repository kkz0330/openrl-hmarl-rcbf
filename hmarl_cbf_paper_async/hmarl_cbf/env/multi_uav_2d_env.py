from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - import-safe fallback
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

from hmarl_cbf.env.lidar import LidarModel
from hmarl_cbf.env.obstacles import (
    copy_obstacle,
    disk_collides_with_obstacle,
    normalize_obstacles,
    obstacle_obstacle_clearance,
    obstacle_surface_distance,
)
from hmarl_cbf.env.observation import ObservationBuilder
from hmarl_cbf.types import AgentState, SafetyMetrics


class MultiUAV2DEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """2D multi-agent goal-reaching environment with obstacle and inter-agent safety."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_agents: int = 4,
        n_obstacles: int = 3,
        world_size: float = 10.0,
        dt: float = 0.1,
        horizon: int = 200,
        action_limit: float = 1.0,
        velocity_limit: float = 2.0,
        agent_radius: float = 0.2,
        goal_threshold: float = 0.3,
        goal_speed_threshold: float = 0.1,
        lidar_beams: int = 32,
        lidar_range: float = 6.0,
        lidar_noise_std: float = 0.0,
        neighbor_radius: float = 4.0,
        max_neighbors: int = 4,
        min_start_goal_separation: float = 1.0,
        min_agent_separation: float = 0.8,
        min_obstacle_clearance: float = 0.4,
        terminate_on_collision: bool = True,
        reward_progress_weight: float = 0.2,
        reward_time_penalty: float = 0.01,
        reward_reach_bonus: float = 1.0,
        reward_collision_penalty: float = 1.0,
        reward_oob_penalty: float = 1.0,
        initial_speed_toward_goal: float = 0.0,
        obstacle_rect_prob: float = 0.0,
        obstacle_circle_radius_min: float = 0.4,
        obstacle_circle_radius_max: float = 1.0,
        obstacle_rect_half_extent_min: float = 0.4,
        obstacle_rect_half_extent_max: float = 1.0,
        obstacle_rect_yaw_max: float = 0.0,
        obstacle_allow_outside_world: bool = False,
        rect_base_margin_extra: float = 0.0,
        rect_corner_margin_enabled: bool = False,
        rect_corner_margin_max: float = 0.0,
        rect_corner_proximity_distance: float = 0.4,
        rect_corner_speed_min: float = 0.05,
        rect_corner_alignment_power: float = 1.0,
        rect_dual_edge_cbf_enabled: bool = False,
        rect_dual_edge_proximity_distance: float = 0.0,
    ) -> None:
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if n_obstacles < 0:
            raise ValueError("n_obstacles must be >= 0")
        self.n_agents = n_agents
        self.n_obstacles = n_obstacles
        self.world_size = float(world_size)
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.action_limit = float(action_limit)
        self.velocity_limit = float(velocity_limit)
        self.agent_radius = float(agent_radius)
        self.goal_threshold = float(goal_threshold)
        self.goal_speed_threshold = float(goal_speed_threshold)
        self.min_start_goal_separation = float(min_start_goal_separation)
        self.min_agent_separation = float(min_agent_separation)
        self.min_obstacle_clearance = float(min_obstacle_clearance)
        self.terminate_on_collision = bool(terminate_on_collision)
        self.reward_progress_weight = float(reward_progress_weight)
        self.reward_time_penalty = float(reward_time_penalty)
        self.reward_reach_bonus = float(reward_reach_bonus)
        self.reward_collision_penalty = float(reward_collision_penalty)
        self.reward_oob_penalty = float(reward_oob_penalty)
        self.initial_speed_toward_goal = float(max(0.0, initial_speed_toward_goal))
        self.obstacle_rect_prob = float(np.clip(obstacle_rect_prob, 0.0, 1.0))
        self.obstacle_circle_radius_min = float(obstacle_circle_radius_min)
        self.obstacle_circle_radius_max = float(obstacle_circle_radius_max)
        self.obstacle_rect_half_extent_min = float(obstacle_rect_half_extent_min)
        self.obstacle_rect_half_extent_max = float(obstacle_rect_half_extent_max)
        self.obstacle_rect_yaw_max = float(max(0.0, obstacle_rect_yaw_max))
        self.obstacle_allow_outside_world = bool(obstacle_allow_outside_world)
        self.rect_base_margin_extra = float(max(0.0, rect_base_margin_extra))
        self.rect_corner_margin_enabled = bool(rect_corner_margin_enabled)
        self.rect_corner_margin_max = float(max(0.0, rect_corner_margin_max))
        self.rect_corner_proximity_distance = float(max(1e-6, rect_corner_proximity_distance))
        self.rect_corner_speed_min = float(max(0.0, rect_corner_speed_min))
        self.rect_corner_alignment_power = float(max(0.25, rect_corner_alignment_power))
        self.rect_dual_edge_cbf_enabled = bool(rect_dual_edge_cbf_enabled)
        self.rect_dual_edge_proximity_distance = float(max(0.0, rect_dual_edge_proximity_distance))
        if self.obstacle_circle_radius_min <= 0.0 or self.obstacle_circle_radius_max < self.obstacle_circle_radius_min:
            raise ValueError("invalid circle obstacle radius range")
        if self.obstacle_rect_half_extent_min <= 0.0 or self.obstacle_rect_half_extent_max < self.obstacle_rect_half_extent_min:
            raise ValueError("invalid rect obstacle half-extent range")
        self.step_count = 0
        self._rng = np.random.default_rng()

        self.lidar = LidarModel(n_beam=lidar_beams, max_range=lidar_range, noise_std=lidar_noise_std)
        self.obs_builder = ObservationBuilder(self.lidar, neighbor_radius=neighbor_radius, max_neighbors=max_neighbors)

        self._states: List[AgentState] = []
        self._obstacles: List[Dict[str, np.ndarray | float]] = []
        self._frozen_agents: set[int] = set()

        if spaces is not None:
            self.action_space = spaces.Box(
                low=-self.action_limit,
                high=self.action_limit,
                shape=(self.n_agents, 2),
                dtype=np.float32,
            )
            low_dim = 4 + 2 + lidar_beams + (max_neighbors * 4)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_agents, low_dim), dtype=np.float32)

    def _sample_point(self, rng: np.random.Generator, margin: float = 0.5) -> np.ndarray:
        low = -self.world_size + margin
        high = self.world_size - margin
        return rng.uniform(low, high, size=(2,)).astype(np.float32)

    def _sample_obstacles(self, rng: np.random.Generator) -> List[Dict[str, np.ndarray | float]]:
        obstacles: List[Dict[str, np.ndarray | float]] = []
        for _ in range(self.n_obstacles):
            placed = False
            for _attempt in range(300):
                if float(rng.uniform()) < self.obstacle_rect_prob:
                    half_extents = rng.uniform(
                        self.obstacle_rect_half_extent_min,
                        self.obstacle_rect_half_extent_max,
                        size=(2,),
                    ).astype(np.float32)
                    margin = 0.0 if self.obstacle_allow_outside_world else float(max(1.0, np.max(half_extents) + 0.2))
                    center = self._sample_point(rng, margin=margin)
                    candidate: Dict[str, np.ndarray | float] = {
                        "type": "rect",
                        "center": center,
                        "half_extents": half_extents,
                        "yaw": float(rng.uniform(-self.obstacle_rect_yaw_max, self.obstacle_rect_yaw_max))
                        if self.obstacle_rect_yaw_max > 0.0
                        else 0.0,
                    }
                else:
                    radius = float(rng.uniform(self.obstacle_circle_radius_min, self.obstacle_circle_radius_max))
                    margin = 0.0 if self.obstacle_allow_outside_world else max(1.0, radius + 0.2)
                    center = self._sample_point(rng, margin=margin)
                    candidate = {"type": "circle", "center": center, "radius": radius}
                if all(
                    obstacle_obstacle_clearance(candidate, item) > self.agent_radius
                    for item in obstacles
                ):
                    obstacles.append(candidate)
                    placed = True
                    break
            if not placed:
                raise RuntimeError("failed to sample non-overlapping obstacles")
        return obstacles

    def _is_point_clear_of_obstacles(self, point: np.ndarray, obstacles: Sequence[Dict[str, np.ndarray | float]]) -> bool:
        for obs in obstacles:
            clearance = self.agent_radius + self.min_obstacle_clearance
            if obstacle_surface_distance(point, obs) <= clearance:
                return False
        return True

    def _sample_states(self, rng: np.random.Generator) -> List[AgentState]:
        states: List[AgentState] = []
        for i in range(self.n_agents):
            placed = False
            for _attempt in range(600):
                position = self._sample_point(rng)
                goal = self._sample_point(rng)
                if np.linalg.norm(goal - position) < self.min_start_goal_separation:
                    continue
                if not self._is_point_clear_of_obstacles(position, self._obstacles):
                    continue
                if not self._is_point_clear_of_obstacles(goal, self._obstacles):
                    continue
                if any(np.linalg.norm(position - state.position) < self.min_agent_separation for state in states):
                    continue
                states.append(
                    AgentState(
                        agent_id=i,
                        position=position,
                        velocity=self._initial_velocity(position=position, goal=goal),
                        goal=goal,
                        radius=self.agent_radius,
                    )
                )
                placed = True
                break
            if not placed:
                raise RuntimeError("failed to sample non-overlapping agent states/goals")
        return states

    def _parse_states_option(self, states_option: Sequence[Dict[str, Any] | AgentState]) -> List[AgentState]:
        parsed: List[AgentState] = []
        if len(states_option) != self.n_agents:
            raise ValueError(f"states option must contain {self.n_agents} entries")
        for idx, item in enumerate(states_option):
            if isinstance(item, AgentState):
                parsed.append(
                    AgentState(
                        agent_id=idx,
                        position=item.position.copy(),
                        velocity=item.velocity.copy(),
                        goal=item.goal.copy(),
                        radius=item.radius,
                    )
                )
                continue
            position = np.asarray(item["position"], dtype=np.float32).reshape(2)
            goal = np.asarray(item["goal"], dtype=np.float32).reshape(2)
            if "velocity" in item:
                velocity = np.asarray(item["velocity"], dtype=np.float32).reshape(2)
            else:
                velocity = self._initial_velocity(position=position, goal=goal)
            radius = float(item.get("radius", self.agent_radius))
            parsed.append(
                AgentState(
                    agent_id=idx,
                    position=position,
                    velocity=velocity,
                    goal=goal,
                    radius=radius,
                )
            )
        return parsed

    def _initial_velocity(self, position: np.ndarray, goal: np.ndarray) -> np.ndarray:
        speed = float(self.initial_speed_toward_goal)
        if speed <= 0.0:
            return np.zeros(2, dtype=np.float32)
        vec = np.asarray(goal - position, dtype=np.float32).reshape(2)
        dist = float(np.linalg.norm(vec))
        if dist <= 1e-6:
            return np.zeros(2, dtype=np.float32)
        direction = vec / dist
        v0 = direction * min(speed, self.velocity_limit)
        return np.asarray(v0, dtype=np.float32).reshape(2)

    def _parse_obstacles_option(self, obstacles_option: Sequence[Dict[str, Any]]) -> List[Dict[str, np.ndarray | float]]:
        return normalize_obstacles(obstacles_option)

    def _out_of_bounds_flags(self) -> Dict[int, bool]:
        flags: Dict[int, bool] = {}
        for state in self._states:
            x, y = state.position
            flags[state.agent_id] = bool(
                (x < -self.world_size) or (x > self.world_size) or (y < -self.world_size) or (y > self.world_size)
            )
        return flags

    def get_agent_states(self) -> List[AgentState]:
        return [
            AgentState(
                agent_id=s.agent_id,
                position=s.position.copy(),
                velocity=s.velocity.copy(),
                goal=s.goal.copy(),
                radius=s.radius,
            )
            for s in self._states
        ]

    def get_obstacles(self) -> List[Dict[str, np.ndarray | float]]:
        return [copy_obstacle(o) for o in self._obstacles]

    def freeze_agents(self, agent_ids: Sequence[int]) -> None:
        for agent_id in agent_ids:
            idx = int(agent_id)
            if 0 <= idx < self.n_agents:
                self._frozen_agents.add(idx)
                self._states[idx].velocity = np.zeros(2, dtype=np.float32)

    def unfreeze_all_agents(self) -> None:
        self._frozen_agents.clear()

    def collision_mask(self) -> Dict[int, bool]:
        flags = {i: False for i in range(self.n_agents)}
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(self._states[i].position - self._states[j].position)
                if dist <= (self._states[i].radius + self._states[j].radius):
                    flags[i] = True
                    flags[j] = True
        for i in range(self.n_agents):
            for obs in self._obstacles:
                if disk_collides_with_obstacle(self._states[i].position, self._states[i].radius, obs):
                    flags[i] = True
        return flags

    def finish_mask(self) -> Dict[int, bool]:
        return {
            s.agent_id: bool(
                (np.linalg.norm(s.goal - s.position) <= self.goal_threshold)
                and (np.linalg.norm(s.velocity) <= self.goal_speed_threshold)
            )
            for s in self._states
        }

    def unsafe_mask(self) -> Dict[int, bool]:
        collision = self.collision_mask()
        oob = self._out_of_bounds_flags()
        return {i: bool(collision[i] or oob[i]) for i in range(self.n_agents)}

    def get_cost(self) -> Dict[int, float]:
        unsafe = self.unsafe_mask()
        return {i: (1.0 if unsafe[i] else 0.0) for i in range(self.n_agents)}

    def _collect_safety_metrics(
        self,
        collision: Dict[int, bool],
        reach: Dict[int, bool],
        unsafe: Dict[int, bool],
    ) -> Dict[int, SafetyMetrics]:
        metrics: Dict[int, SafetyMetrics] = {}
        for state in self._states:
            h_agent = np.inf
            for other in self._states:
                if state.agent_id == other.agent_id:
                    continue
                h_agent = min(
                    h_agent,
                    float(np.dot(state.position - other.position, state.position - other.position) - (2.0 * self.agent_radius) ** 2),
                )
            h_obs = np.inf
            for obs in self._obstacles:
                clearance = obstacle_surface_distance(state.position, obs) - self.agent_radius
                h_obs = min(h_obs, float(clearance))
            metrics[state.agent_id] = SafetyMetrics(
                min_h_agent=float(h_agent if np.isfinite(h_agent) else 0.0),
                min_h_obstacle=float(h_obs if np.isfinite(h_obs) else 0.0),
                collision=unsafe[state.agent_id],
                reached_goal=reach[state.agent_id],
                qp_feasible=True,
            )
        return metrics

    def reset(self, *, seed: int | None = None, options: Dict[str, object] | None = None):  # type: ignore[override]
        options = options or {}
        self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self._frozen_agents.clear()
        if "obstacles" in options:
            self._obstacles = self._parse_obstacles_option(options["obstacles"])  # type: ignore[arg-type]
        else:
            self._obstacles = self._sample_obstacles(self._rng)
        if "states" in options:
            self._states = self._parse_states_option(options["states"])  # type: ignore[arg-type]
        else:
            self._states = self._sample_states(self._rng)
        obs = self.obs_builder.build(self._states, self._obstacles)
        info = {
            "seed": seed,
            "step_count": self.step_count,
            "collision_flags": self.collision_mask(),
            "reach_flags": self.finish_mask(),
            "unsafe_flags": self.unsafe_mask(),
            "costs": self.get_cost(),
        }
        return obs, info

    def step(self, actions: Dict[int, np.ndarray]):  # type: ignore[override]
        self.step_count += 1
        prev_dist = {
            state.agent_id: float(np.linalg.norm(state.goal - state.position))
            for state in self._states
        }

        for state in self._states:
            if state.agent_id in self._frozen_agents:
                state.velocity = np.zeros(2, dtype=np.float32)
                continue
            action = np.asarray(actions.get(state.agent_id, np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(2)
            action = np.clip(action, -self.action_limit, self.action_limit)
            state.velocity = np.clip(state.velocity + action * self.dt, -self.velocity_limit, self.velocity_limit)
            state.position = state.position + state.velocity * self.dt

        collision = self.collision_mask()
        reach = self.finish_mask()
        unsafe = self.unsafe_mask()
        out_of_bounds = self._out_of_bounds_flags()
        costs = self.get_cost()

        rewards: Dict[int, float] = {}
        for i in range(self.n_agents):
            state = self._states[i]
            curr_dist = float(np.linalg.norm(state.goal - state.position))
            progress = prev_dist[i] - curr_dist
            reward = self.reward_progress_weight * progress - self.reward_time_penalty
            if reach[i]:
                reward += self.reward_reach_bonus
            if collision[i]:
                reward -= self.reward_collision_penalty
            if out_of_bounds[i]:
                reward -= self.reward_oob_penalty
            rewards[i] = float(reward)

        terminated = bool(all(reach.values()) or (self.terminate_on_collision and any(unsafe.values())))
        truncated = bool(self.step_count >= self.horizon)
        obs = self.obs_builder.build(self._states, self._obstacles)
        metrics = self._collect_safety_metrics(collision, reach, unsafe)
        info = {
            "collision_flags": collision,
            "reach_flags": reach,
            "unsafe_flags": unsafe,
            "out_of_bounds_flags": out_of_bounds,
            "costs": costs,
            "safety_metrics": metrics,
            "step_count": self.step_count,
            "frozen_agents": sorted(int(i) for i in self._frozen_agents),
        }
        return obs, rewards, terminated, truncated, info
