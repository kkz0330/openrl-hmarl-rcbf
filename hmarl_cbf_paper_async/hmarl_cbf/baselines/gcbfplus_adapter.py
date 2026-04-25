from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from hmarl_cbf.types import AgentObsLow, AgentState, LIDAR_HIT_OBSTACLE


def _ensure_gcbfplus_on_path() -> None:
    root = Path(__file__).resolve().parents[4]
    pkg_root = root / "_ext" / "gcbfplus"
    pkg_str = str(pkg_root)
    if pkg_root.exists() and pkg_str not in sys.path:
        sys.path.insert(0, pkg_str)


@dataclass(slots=True)
class GCBFPlusHandcraftedConfig:
    alpha: float = 1.0
    k: int = 3
    action_limit: float = 2.0
    velocity_limit: float = 2.0
    comm_radius: float = 3.0
    car_radius: float = 0.05
    n_rays: int = 32
    mass: float = 1.0
    dt: float = 0.03
    world_size: float = 10.0
    horizon: int = 600

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "GCBFPlusHandcraftedConfig":
        return GCBFPlusHandcraftedConfig(
            alpha=float(data.get("alpha", 1.0)),
            k=int(data.get("k", 3)),
            action_limit=float(data.get("action_limit", 2.0)),
            velocity_limit=float(data.get("velocity_limit", 2.0)),
            comm_radius=float(data.get("comm_radius", 3.0)),
            car_radius=float(data.get("car_radius", 0.05)),
            n_rays=int(data.get("n_rays", 32)),
            mass=float(data.get("mass", 1.0)),
            dt=float(data.get("dt", 0.03)),
            world_size=float(data.get("world_size", 10.0)),
            horizon=int(data.get("horizon", 600)),
        )


@dataclass(slots=True)
class _GraphBuffers:
    n_agents: int
    n_hits: int
    n_nodes_unpadded: int
    n_nodes_padded: int
    pad_id: int
    agent: np.ndarray
    goal: np.ndarray
    lidar: np.ndarray
    states_padded: np.ndarray
    nodes_jnp: Any
    node_type_jnp: Any
    n_node_jnp: Any
    n_edge_jnp: Any
    dummy_edges_jnp: Any
    dummy_receivers_jnp: Any
    dummy_senders_jnp: Any


class HmarlGCBFDoubleIntegratorAdapter:
    def __init__(self, config: GCBFPlusHandcraftedConfig) -> None:
        _ensure_gcbfplus_on_path()
        from gcbfplus.env.double_integrator import DoubleIntegrator
        from gcbfplus.utils.graph import GraphsTuple
        import jax.numpy as jnp

        class _Adapter(DoubleIntegrator):
            def __init__(self, cfg: GCBFPlusHandcraftedConfig) -> None:
                params = dict(DoubleIntegrator.PARAMS)
                params["car_radius"] = float(cfg.car_radius)
                params["comm_radius"] = float(cfg.comm_radius)
                params["n_rays"] = int(cfg.n_rays)
                params["n_obs"] = 0
                params["m"] = float(cfg.mass)
                super().__init__(
                    num_agents=0,  # placeholder, overwritten below
                    area_size=float(cfg.world_size * 2.0),
                    max_step=int(cfg.horizon),
                    dt=float(cfg.dt),
                    params=params,
                )
                self._num_agents = 0
                self._action_limit = float(cfg.action_limit)
                self._velocity_limit = float(cfg.velocity_limit)

            def configure_num_agents(self, n_agents: int) -> None:
                self._num_agents = int(n_agents)

            def action_lim(self):
                lower = jnp.full((2,), -self._action_limit, dtype=jnp.float32)
                upper = jnp.full((2,), self._action_limit, dtype=jnp.float32)
                return lower, upper

            def state_lim(self, state=None):
                lower = jnp.asarray(
                    [-jnp.inf, -jnp.inf, -self._velocity_limit, -self._velocity_limit],
                    dtype=jnp.float32,
                )
                upper = jnp.asarray(
                    [jnp.inf, jnp.inf, self._velocity_limit, self._velocity_limit],
                    dtype=jnp.float32,
                )
                return lower, upper

            def u_ref(self, graph):
                agent = graph.type_states(type_idx=0, n_type=self.num_agents)
                goal = graph.type_states(type_idx=1, n_type=self.num_agents)
                error = goal - agent
                norm = jnp.linalg.norm(error, axis=-1, keepdims=True)
                safe_norm = jnp.maximum(norm, 1e-6)
                error_max = jnp.abs(error / safe_norm * self._params["comm_radius"])
                error = jnp.clip(error, -error_max, error_max)
                return self.clip_action(error @ self._K.T)

        self._jnp = jnp
        self._GraphsTuple = GraphsTuple
        self._env = _Adapter(config)
        self.config = config
        self._buffers_by_agent_count: dict[int, _GraphBuffers] = {}

    @property
    def env(self):
        return self._env

    def _get_buffers(self, n_agents: int) -> _GraphBuffers:
        cached = self._buffers_by_agent_count.get(n_agents)
        if cached is not None:
            return cached

        n_hits = int(self.config.n_rays) * n_agents
        n_nodes_unpadded = 2 * n_agents + n_hits
        n_nodes_padded = n_nodes_unpadded + 1
        pad_id = n_nodes_unpadded

        node_feats = np.zeros((n_nodes_padded, 3), dtype=np.float32)
        node_feats[:n_agents, 2] = 1.0
        node_feats[n_agents : 2 * n_agents, 1] = 1.0
        node_feats[2 * n_agents : 2 * n_agents + n_hits, 0] = 1.0

        node_type = np.full((n_nodes_padded,), -1, dtype=np.int32)
        node_type[:n_agents] = int(self._env.AGENT)
        node_type[n_agents : 2 * n_agents] = int(self._env.GOAL)
        node_type[2 * n_agents : 2 * n_agents + n_hits] = int(self._env.OBS)

        buffers = _GraphBuffers(
            n_agents=n_agents,
            n_hits=n_hits,
            n_nodes_unpadded=n_nodes_unpadded,
            n_nodes_padded=n_nodes_padded,
            pad_id=pad_id,
            agent=np.zeros((n_agents, self._env.state_dim), dtype=np.float32),
            goal=np.zeros((n_agents, self._env.state_dim), dtype=np.float32),
            lidar=np.zeros((n_hits, self._env.state_dim), dtype=np.float32),
            states_padded=np.full((n_nodes_padded, self._env.state_dim), -1.0, dtype=np.float32),
            nodes_jnp=self._jnp.asarray(node_feats, dtype=self._jnp.float32),
            node_type_jnp=self._jnp.asarray(node_type, dtype=self._jnp.int32),
            n_node_jnp=self._jnp.asarray(n_nodes_padded, dtype=self._jnp.int32),
            n_edge_jnp=self._jnp.asarray(1, dtype=self._jnp.int32),
            dummy_edges_jnp=self._jnp.zeros((1, self._env.edge_dim), dtype=self._jnp.float32),
            dummy_receivers_jnp=self._jnp.asarray([pad_id], dtype=self._jnp.int32),
            dummy_senders_jnp=self._jnp.asarray([pad_id], dtype=self._jnp.int32),
        )
        self._buffers_by_agent_count[n_agents] = buffers
        return buffers

    def graph_from_hmarl(
        self,
        states: Mapping[int, AgentState],
        obs_low: Mapping[int, AgentObsLow],
    ):
        agent_ids = sorted(states.keys())
        self._env.configure_num_agents(len(agent_ids))
        n_agents = len(agent_ids)
        buffers = self._get_buffers(n_agents)
        n_rays = int(self.config.n_rays)

        agent = buffers.agent
        goal = buffers.goal
        lidar = buffers.lidar

        for agent_idx, aid in enumerate(agent_ids):
            state = states[aid]
            agent[agent_idx, :2] = np.asarray(state.position, dtype=np.float32).reshape(2)
            agent[agent_idx, 2:] = np.asarray(state.velocity, dtype=np.float32).reshape(2)
            goal[agent_idx, :2] = np.asarray(state.goal, dtype=np.float32).reshape(2)
            goal[agent_idx, 2:] = 0.0

            scan = obs_low[aid].lidar_scan
            hit_slice = slice(agent_idx * n_rays, (agent_idx + 1) * n_rays)
            lidar[hit_slice, :2] = np.asarray(scan.hit_points, dtype=np.float32).reshape(n_rays, 2)
            lidar[hit_slice, 2:] = 0.0

            hit_kinds = np.asarray(scan.hit_kinds, dtype=np.int32).reshape(n_rays)
            non_obstacle = hit_kinds != int(LIDAR_HIT_OBSTACLE)
            if np.any(non_obstacle):
                angles = np.asarray(scan.angles, dtype=np.float32).reshape(n_rays)
                origin = np.asarray(scan.origin, dtype=np.float32).reshape(1, 2)
                dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
                far_points = origin + float(scan.max_range) * dirs
                lidar_points = lidar[hit_slice, :2]
                lidar_points[non_obstacle] = far_points[non_obstacle]

        states_padded = buffers.states_padded
        states_padded[:n_agents] = agent
        states_padded[n_agents : 2 * n_agents] = goal
        states_padded[2 * n_agents : 2 * n_agents + buffers.n_hits] = lidar
        states_padded[buffers.pad_id] = -1.0

        env_state = self._env.EnvState(
            self._jnp.asarray(agent, dtype=self._jnp.float32),
            self._jnp.asarray(goal, dtype=self._jnp.float32),
            None,
        )
        return self._GraphsTuple(
            buffers.n_node_jnp,
            buffers.n_edge_jnp,
            buffers.nodes_jnp,
            buffers.dummy_edges_jnp,
            self._jnp.asarray(states_padded, dtype=self._jnp.float32),
            buffers.dummy_receivers_jnp,
            buffers.dummy_senders_jnp,
            buffers.node_type_jnp,
            env_state,
            None,
        )


class GCBFPlusHandcraftedController:
    def __init__(self, config: GCBFPlusHandcraftedConfig) -> None:
        _ensure_gcbfplus_on_path()
        from gcbfplus.algo.dec_share_cbf import DecShareCBF
        from gcbfplus.algo.utils import get_pwise_cbf_fn

        self.adapter = HmarlGCBFDoubleIntegratorAdapter(config)
        env = self.adapter.env
        self.controller = DecShareCBF(
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=0,
            alpha=float(config.alpha),
        )
        self._get_pwise_cbf_fn = get_pwise_cbf_fn
        self.config = config
        self._debug_first_call = True

    def _ensure_agent_count(self, n_agents: int) -> None:
        self.adapter.env.configure_num_agents(n_agents)
        self.controller._n_agents = int(n_agents)
        self.controller.k = int(max(1, self.config.k))
        self.controller.cbf = self._get_pwise_cbf_fn(self.adapter.env, self.controller.k)

    def act(
        self,
        states: Mapping[int, AgentState],
        obs_low: Mapping[int, AgentObsLow],
    ) -> tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        agent_ids = sorted(states.keys())
        self._ensure_agent_count(len(agent_ids))
        if self._debug_first_call:
            print("[debug] gcbf-adapter: graph-build-start", flush=True)
        graph = self.adapter.graph_from_hmarl(states=states, obs_low=obs_low)
        if self._debug_first_call:
            print("[debug] gcbf-adapter: graph-build-done", flush=True)
            print("[debug] gcbf-adapter: qp-action-start", flush=True)
        actions, relax = self.controller.get_qp_action(graph)
        if self._debug_first_call:
            print("[debug] gcbf-adapter: qp-action-done", flush=True)
            self._debug_first_call = False
        actions_np = np.asarray(actions, dtype=np.float32)
        relax_np = np.asarray(relax, dtype=np.float32)
        action_map = {aid: actions_np[idx].reshape(2).astype(np.float32) for idx, aid in enumerate(agent_ids)}
        relax_map = {aid: relax_np[idx].reshape(-1).astype(np.float32) for idx, aid in enumerate(agent_ids)}
        return action_map, relax_map
