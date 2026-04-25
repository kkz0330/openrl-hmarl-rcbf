from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.env.obstacles import normalize_obstacle
from hmarl_cbf.types import AgentObsHigh, AgentObsLow, AgentState


def _sorted_agent_ids_from_mapping(mapping: Mapping[int, Any]) -> List[int]:
    return sorted(int(agent_id) for agent_id in mapping.keys())


def _ensure_vec(value: np.ndarray | Sequence[float], *, dtype: np.dtype[np.floating[Any]] = np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


@dataclass(slots=True)
class CoreEnvStep:
    obs: Dict[int, Dict[str, AgentObsHigh | AgentObsLow]]
    rewards: Dict[int, float]
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class CoreEnv:
    """Thin stateful wrapper around ``MultiUAV2DEnv`` for OpenRL-facing rebuild work.

    This layer keeps the existing environment logic unchanged while exposing:
    - atomic reset/step access
    - cached state/observation/info snapshots
    - stable tensor-style views for high-level actor/critic
    - stable tensor-style views for low-level actor/critic
    """

    _AGENT_STATE_DIM = 7
    _OBSTACLE_DIM = 6

    def __init__(
        self,
        env: MultiUAV2DEnv,
        *,
        obstacle_slots: int | None = None,
    ) -> None:
        self.env = env
        self.obstacle_slots = int(max(0, obstacle_slots if obstacle_slots is not None else env.n_obstacles))

        self._last_obs: Dict[int, Dict[str, AgentObsHigh | AgentObsLow]] = {}
        self._last_info: Dict[str, Any] = {}
        self._last_rewards: Dict[int, float] = {}
        self._last_terminated: bool = False
        self._last_truncated: bool = False

    @classmethod
    def from_env_config(
        cls,
        env_cfg: Mapping[str, Any],
        *,
        obstacle_slots: int | None = None,
    ) -> "CoreEnv":
        env = MultiUAV2DEnv(**dict(env_cfg))
        return cls(env=env, obstacle_slots=obstacle_slots)

    @property
    def agent_ids(self) -> List[int]:
        return [state.agent_id for state in self.env.get_agent_states()]

    @property
    def last_obs(self) -> Dict[int, Dict[str, AgentObsHigh | AgentObsLow]]:
        return self._last_obs

    @property
    def last_info(self) -> Dict[str, Any]:
        return self._last_info

    @property
    def last_rewards(self) -> Dict[int, float]:
        return self._last_rewards

    @property
    def last_terminated(self) -> bool:
        return self._last_terminated

    @property
    def last_truncated(self) -> bool:
        return self._last_truncated

    def reset_scene(
        self,
        *,
        seed: int | None = None,
        states: Sequence[Dict[str, Any]] | Sequence[AgentState] | None = None,
        obstacles: Sequence[Dict[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[int, Dict[str, AgentObsHigh | AgentObsLow]], Dict[str, Any]]:
        reset_options: Dict[str, Any] = dict(options or {})
        if states is not None:
            reset_options["states"] = list(states)
        if obstacles is not None:
            reset_options["obstacles"] = list(obstacles)

        obs, info = self.env.reset(seed=seed, options=reset_options)
        self._last_obs = obs
        self._last_info = info
        self._last_rewards = {int(agent_id): 0.0 for agent_id in _sorted_agent_ids_from_mapping(obs)}
        self._last_terminated = False
        self._last_truncated = False
        return obs, info

    def step_low_level(self, actions: Mapping[int, np.ndarray]) -> CoreEnvStep:
        obs, rewards, terminated, truncated, info = self.env.step(dict(actions))
        self._last_obs = obs
        self._last_info = info
        self._last_rewards = {int(agent_id): float(reward) for agent_id, reward in rewards.items()}
        self._last_terminated = bool(terminated)
        self._last_truncated = bool(truncated)
        return CoreEnvStep(
            obs=obs,
            rewards=self._last_rewards,
            terminated=self._last_terminated,
            truncated=self._last_truncated,
            info=info,
        )

    def get_states(self) -> Dict[int, AgentState]:
        return {state.agent_id: state for state in self.env.get_agent_states()}

    def get_obstacles(self) -> List[Dict[str, Any]]:
        return self.env.get_obstacles()

    def get_obs_high(self) -> Dict[int, AgentObsHigh]:
        return {int(agent_id): bundle["high"] for agent_id, bundle in self._last_obs.items()}

    def get_obs_low(self) -> Dict[int, AgentObsLow]:
        return {int(agent_id): bundle["low"] for agent_id, bundle in self._last_obs.items()}

    @classmethod
    def flatten_high_obs(cls, obs_high: AgentObsHigh) -> np.ndarray:
        return np.concatenate(
            [
                _ensure_vec(obs_high.self_state),
                _ensure_vec(obs_high.goal_relative),
                _ensure_vec(obs_high.neighbor_summary),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    @classmethod
    def flatten_low_obs(cls, obs_low: AgentObsLow) -> np.ndarray:
        return np.asarray(obs_low.flat, dtype=np.float32).reshape(-1)

    @classmethod
    def flatten_agent_state(cls, state: AgentState) -> np.ndarray:
        return np.concatenate(
            [
                _ensure_vec(state.position),
                _ensure_vec(state.velocity),
                _ensure_vec(state.goal),
                np.asarray([float(state.radius)], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    @classmethod
    def flatten_obstacle(cls, obstacle: Mapping[str, Any]) -> np.ndarray:
        obs = normalize_obstacle(dict(obstacle))
        center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
        obstacle_type = str(obs["type"])
        if obstacle_type in ("circle", "point", "lidar_point", "lidar_circle"):
            return np.asarray(
                [0.0, center[0], center[1], float(obs["radius"]), 0.0, 0.0],
                dtype=np.float32,
            )
        if obstacle_type == "lidar_line":
            start = np.asarray(obs["start"], dtype=np.float32).reshape(2)
            end = np.asarray(obs["end"], dtype=np.float32).reshape(2)
            length = float(np.linalg.norm(end - start))
            yaw = float(np.arctan2(end[1] - start[1], end[0] - start[0]))
            return np.asarray([2.0, center[0], center[1], length, 0.0, yaw], dtype=np.float32)
        half_extents = np.asarray(obs["half_extents"], dtype=np.float32).reshape(2)
        yaw = float(obs.get("yaw", 0.0))
        return np.asarray([1.0, center[0], center[1], half_extents[0], half_extents[1], yaw], dtype=np.float32)

    def _ordered_states(self) -> List[AgentState]:
        states = self.env.get_agent_states()
        states.sort(key=lambda item: int(item.agent_id))
        return states

    def _ordered_high_obs(self) -> List[AgentObsHigh]:
        obs_high = self.get_obs_high()
        return [obs_high[agent_id] for agent_id in _sorted_agent_ids_from_mapping(obs_high)]

    def _ordered_low_obs(self) -> List[AgentObsLow]:
        obs_low = self.get_obs_low()
        return [obs_low[agent_id] for agent_id in _sorted_agent_ids_from_mapping(obs_low)]

    def build_high_actor_obs(self) -> Dict[int, np.ndarray]:
        return {
            int(agent_id): self.flatten_high_obs(obs_high)
            for agent_id, obs_high in self.get_obs_high().items()
        }

    def build_high_critic_obs(self) -> np.ndarray:
        states = self._ordered_states()
        obstacles = self.get_obstacles()

        agent_block = np.zeros((self.env.n_agents, self._AGENT_STATE_DIM), dtype=np.float32)
        for row, state in enumerate(states):
            agent_block[row] = self.flatten_agent_state(state)

        obstacle_block = np.zeros((self.obstacle_slots, self._OBSTACLE_DIM), dtype=np.float32)
        for row, obstacle in enumerate(obstacles[: self.obstacle_slots]):
            obstacle_block[row] = self.flatten_obstacle(obstacle)

        wind = np.asarray(self.env.get_current_wind_accel(), dtype=np.float32).reshape(2)
        step_meta = np.asarray(
            [
                float(getattr(self.env, "step_count", 0)),
                float(getattr(self.env, "horizon", 0)),
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                agent_block.reshape(-1),
                obstacle_block.reshape(-1),
                wind,
                step_meta,
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    def build_low_actor_obs(
        self,
        *,
        skill_ids: Mapping[int, int] | None = None,
        n_skills: int | None = None,
    ) -> Dict[int, np.ndarray]:
        obs_low = self.get_obs_low()
        result: Dict[int, np.ndarray] = {}
        use_skill = skill_ids is not None and n_skills is not None and int(n_skills) > 0

        for agent_id, low in obs_low.items():
            flat = self.flatten_low_obs(low)
            if use_skill:
                one_hot = np.zeros(int(n_skills), dtype=np.float32)
                skill_id = int(skill_ids.get(agent_id, 0))
                if 0 <= skill_id < int(n_skills):
                    one_hot[skill_id] = 1.0
                flat = np.concatenate([flat, one_hot], axis=0)
            result[int(agent_id)] = flat.astype(np.float32, copy=False)
        return result

    def build_low_critic_obs(
        self,
        *,
        skill_ids: Mapping[int, int] | None = None,
        n_skills: int | None = None,
        include_local_low_obs: bool = False,
    ) -> np.ndarray:
        global_state = self.build_high_critic_obs()
        blocks: List[np.ndarray] = [global_state]

        if skill_ids is not None and n_skills is not None and int(n_skills) > 0:
            skill_block = np.zeros((self.env.n_agents, int(n_skills)), dtype=np.float32)
            for row, agent_id in enumerate(self.agent_ids):
                skill_id = int(skill_ids.get(agent_id, 0))
                if 0 <= skill_id < int(n_skills) and row < self.env.n_agents:
                    skill_block[row, skill_id] = 1.0
            blocks.append(skill_block.reshape(-1))

        if include_local_low_obs:
            local_obs = self.build_low_actor_obs(skill_ids=skill_ids, n_skills=n_skills)
            ordered = [local_obs[agent_id] for agent_id in sorted(local_obs.keys())]
            if ordered:
                blocks.append(np.concatenate(ordered, axis=0).astype(np.float32, copy=False))

        return np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
