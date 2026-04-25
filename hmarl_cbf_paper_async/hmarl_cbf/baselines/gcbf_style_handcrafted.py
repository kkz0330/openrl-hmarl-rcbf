from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from hmarl_cbf.control import DifferentiableQPSolver
from hmarl_cbf.types import AgentObsLow, AgentState, LIDAR_HIT_OBSTACLE, QPProblem, QPSolution


@dataclass(slots=True)
class GCBFStyleHandcraftedConfig:
    alpha: float = 1.0
    k: int = 3
    action_limit: float = 2.0
    velocity_limit: float = 2.0
    comm_radius: float = 3.0
    car_radius: float = 0.05
    n_rays: int = 32
    mass: float = 1.0
    dt: float = 0.03
    q_pos: float = 5.0
    q_vel: float = 5.0
    r_input: float = 1.0
    relax_penalty: float = 1e3
    cbf_slack_max: float = 1e6
    use_stub_if_unavailable: bool = True
    ecos_max_iters: int = 500
    scs_max_iters: int = 10000
    scs_eps: float = 1e-4

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "GCBFStyleHandcraftedConfig":
        return GCBFStyleHandcraftedConfig(
            alpha=float(data.get("alpha", 1.0)),
            k=int(data.get("k", 3)),
            action_limit=float(data.get("action_limit", 2.0)),
            velocity_limit=float(data.get("velocity_limit", 2.0)),
            comm_radius=float(data.get("comm_radius", 3.0)),
            car_radius=float(data.get("car_radius", 0.05)),
            n_rays=int(data.get("n_rays", 32)),
            mass=float(data.get("mass", 1.0)),
            dt=float(data.get("dt", 0.03)),
            q_pos=float(data.get("q_pos", 5.0)),
            q_vel=float(data.get("q_vel", 5.0)),
            r_input=float(data.get("r_input", 1.0)),
            relax_penalty=float(data.get("relax_penalty", 1e3)),
            cbf_slack_max=float(data.get("cbf_slack_max", 1e6)),
            use_stub_if_unavailable=bool(data.get("use_stub_if_unavailable", True)),
            ecos_max_iters=int(data.get("ecos_max_iters", 500)),
            scs_max_iters=int(data.get("scs_max_iters", 10000)),
            scs_eps=float(data.get("scs_eps", 1e-4)),
        )


class GCBFStyleHandcraftedController:
    """
    GCBF-style handcrafted CBF-QP baseline implemented directly on HMARL state/LiDAR.

    The controller mirrors the original DoubleIntegrator handcrafted baseline:
    - Discrete-time LQR nominal u_ref toward [goal_pos, 0, 0]
    - Unified nearest-k pointwise objects across neighbors + LiDAR hit points
    - Per-agent QP with agent-agent responsibility 0.5 and obstacle responsibility 1.0
    """

    def __init__(self, config: GCBFStyleHandcraftedConfig) -> None:
        self.config = config
        self._lqr_gain_cache: np.ndarray | None = None
        self.qp_solver = DifferentiableQPSolver(
            action_dim=2,
            use_stub_if_unavailable=bool(config.use_stub_if_unavailable),
            ecos_max_iters=int(config.ecos_max_iters),
            scs_max_iters=int(config.scs_max_iters),
            scs_eps=float(config.scs_eps),
        )

    def _get_lqr_gain(self) -> np.ndarray:
        if self._lqr_gain_cache is not None:
            return self._lqr_gain_cache

        dt = float(max(1e-6, self.config.dt))
        inv_m = 1.0 / float(max(1e-6, self.config.mass))
        A = np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        B = (dt * np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [inv_m, 0.0],
                [0.0, inv_m],
            ],
            dtype=np.float32,
        )).astype(np.float32)
        Q = np.diag(
            np.asarray(
                [self.config.q_pos, self.config.q_pos, self.config.q_vel, self.config.q_vel],
                dtype=np.float32,
            )
        ).astype(np.float32)
        R = np.diag(np.asarray([self.config.r_input, self.config.r_input], dtype=np.float32)).astype(np.float32)

        P = Q.copy()
        gain = np.zeros((2, 4), dtype=np.float32)
        for _ in range(512):
            lhs = R + B.T @ P @ B
            rhs = B.T @ P @ A
            gain_next = np.linalg.solve(lhs, rhs).astype(np.float32)
            P_next = (Q + A.T @ P @ A - A.T @ P @ B @ gain_next).astype(np.float32)
            if float(np.max(np.abs(P_next - P))) <= 1e-6:
                gain = gain_next
                break
            P = P_next
            gain = gain_next
        self._lqr_gain_cache = gain.astype(np.float32)
        return self._lqr_gain_cache

    def _u_ref_batch(self, goal_states: np.ndarray, agent_states: np.ndarray) -> np.ndarray:
        K = self._get_lqr_gain()
        error = (goal_states - agent_states).astype(np.float32)
        norm = np.linalg.norm(error, axis=1, keepdims=True).astype(np.float32)
        safe_norm = np.maximum(norm, 1e-6).astype(np.float32)
        error_max = np.abs(error / safe_norm * float(self.config.comm_radius)).astype(np.float32)
        error = np.clip(error, -error_max, error_max).astype(np.float32)
        u_ref = (error @ K.T).astype(np.float32)
        u_lim = float(max(1e-6, self.config.action_limit))
        return np.clip(u_ref, -u_lim, u_lim).astype(np.float32)

    def _agent_state_array(self, state: AgentState) -> np.ndarray:
        return np.asarray(
            [state.position[0], state.position[1], state.velocity[0], state.velocity[1]],
            dtype=np.float32,
        )

    def _build_batch_inputs(
        self,
        states: Mapping[int, AgentState],
        obs_low: Mapping[int, AgentObsLow],
    ) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
        agent_ids = sorted(states.keys())
        n_agents = len(agent_ids)
        n_rays = int(self.config.n_rays)

        agent_states = np.stack([self._agent_state_array(states[aid]) for aid in agent_ids], axis=0).astype(np.float32)
        goal_states = np.stack(
            [np.asarray([states[aid].goal[0], states[aid].goal[1], 0.0, 0.0], dtype=np.float32) for aid in agent_ids],
            axis=0,
        ).astype(np.float32)

        hit_points = np.stack(
            [np.asarray(obs_low[aid].lidar_scan.hit_points, dtype=np.float32).reshape(n_rays, 2) for aid in agent_ids],
            axis=0,
        ).astype(np.float32)
        hit_kinds = np.stack(
            [np.asarray(obs_low[aid].lidar_scan.hit_kinds, dtype=np.int32).reshape(n_rays) for aid in agent_ids],
            axis=0,
        )
        angles = np.stack(
            [np.asarray(obs_low[aid].lidar_scan.angles, dtype=np.float32).reshape(n_rays) for aid in agent_ids],
            axis=0,
        ).astype(np.float32)
        origins = np.stack(
            [np.asarray(obs_low[aid].lidar_scan.origin, dtype=np.float32).reshape(2) for aid in agent_ids],
            axis=0,
        ).astype(np.float32)
        max_ranges = np.asarray([float(obs_low[aid].lidar_scan.max_range) for aid in agent_ids], dtype=np.float32).reshape(n_agents, 1, 1)

        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
        far_points = origins[:, None, :] + max_ranges * dirs
        obstacle_mask = (hit_kinds == int(LIDAR_HIT_OBSTACLE)).reshape(n_agents, n_rays, 1)
        hit_points = np.where(obstacle_mask, hit_points, far_points).astype(np.float32)
        lidar_states = np.concatenate([hit_points, np.zeros((n_agents, n_rays, 2), dtype=np.float32)], axis=-1).astype(np.float32)
        return agent_ids, agent_states, goal_states, lidar_states

    def _pairwise_terms_batch(
        self,
        agent_states: np.ndarray,
        lidar_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_agents = int(agent_states.shape[0])
        n_rays = int(lidar_states.shape[1])
        k = int(max(1, min(self.config.k, n_agents + n_rays)))

        agent_block = np.broadcast_to(agent_states[None, :, :], (n_agents, n_agents, 4)).astype(np.float32)
        all_states = np.concatenate([agent_block, lidar_states], axis=1).astype(np.float32)
        pos = agent_states[:, :2].astype(np.float32)
        vel = agent_states[:, 2:].astype(np.float32)
        all_pos = all_states[:, :, :2].astype(np.float32)
        all_vel = all_states[:, :, 2:].astype(np.float32)

        diff = pos[:, None, :] - all_pos
        dist_sq = np.sum(diff * diff, axis=-1).astype(np.float32)
        dist_sq[np.arange(n_agents), np.arange(n_agents)] = 1e6

        nearest_idx = np.argsort(dist_sq, axis=1)[:, :k].astype(np.int32)
        is_obs = (nearest_idx >= n_agents)

        gather_idx_xy = nearest_idx[:, :, None]
        nearest_pos = np.take_along_axis(all_pos, gather_idx_xy.repeat(2, axis=2), axis=1).astype(np.float32)
        nearest_vel = np.take_along_axis(all_vel, gather_idx_xy.repeat(2, axis=2), axis=1).astype(np.float32)
        nearest_dist_sq = np.take_along_axis(dist_sq, nearest_idx, axis=1).astype(np.float32)

        xdiff = (pos[:, None, :] - nearest_pos).astype(np.float32)
        vdiff = (vel[:, None, :] - nearest_vel).astype(np.float32)
        h0 = (nearest_dist_sq - 4.0 * float(self.config.car_radius) ** 2).astype(np.float32)
        h1 = (2.0 * np.sum(xdiff * vdiff, axis=-1) + 10.0 * h0).astype(np.float32)
        lf_h = (2.0 * np.sum(vdiff * vdiff, axis=-1) + 20.0 * np.sum(xdiff * vdiff, axis=-1)).astype(np.float32)
        lg_h = (2.0 * xdiff / float(max(1e-6, self.config.mass))).astype(np.float32)
        responsibility = np.where(is_obs, 1.0, 0.5).astype(np.float32)
        return h1, is_obs, lf_h, lg_h, responsibility

    def _build_qp_problem(
        self,
        u_ref: np.ndarray,
        h: np.ndarray,
        lf_h: np.ndarray,
        lg_h: np.ndarray,
        responsibility: np.ndarray,
    ) -> QPProblem:
        k = int(h.shape[0])
        A_cbf = (-np.asarray(lg_h, dtype=np.float32)).reshape(k, 2)
        b_cbf = (
            np.asarray(responsibility, dtype=np.float32)
            * (
                np.asarray(lf_h, dtype=np.float32)
                + float(self.config.alpha) * np.asarray(h, dtype=np.float32)
            )
        ).astype(np.float32)
        u_lim = float(self.config.action_limit)
        return QPProblem(
            H_mat=np.eye(2, dtype=np.float32),
            f_lin=-np.asarray(u_ref, dtype=np.float32).reshape(2),
            w_clf=np.asarray([0.0], dtype=np.float32),
            w_cbf=np.asarray([float(self.config.relax_penalty)], dtype=np.float32),
            cbf_slack_max=np.asarray([float(self.config.cbf_slack_max)], dtype=np.float32),
            A_cbf=A_cbf,
            b_cbf=b_cbf,
            A_clf=np.zeros((0, 2), dtype=np.float32),
            b_clf=np.zeros((0,), dtype=np.float32),
            u_min=np.asarray([-u_lim, -u_lim], dtype=np.float32),
            u_max=np.asarray([u_lim, u_lim], dtype=np.float32),
            delta_min=0.0,
        )

    def _solve_single(
        self,
        u_ref: np.ndarray,
        h: np.ndarray,
        lf_h: np.ndarray,
        lg_h: np.ndarray,
        responsibility: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        problem = self._build_qp_problem(u_ref, h, lf_h, lg_h, responsibility)
        solution: QPSolution = self.qp_solver.solve(problem)
        action = np.asarray(solution.action, dtype=np.float32).reshape(2)
        cbf_slack = (
            np.asarray(solution.cbf_slack, dtype=np.float32).reshape(-1)
            if solution.cbf_slack is not None
            else np.zeros((h.shape[0],), dtype=np.float32)
        )
        return action, cbf_slack

    def act(
        self,
        states: Mapping[int, AgentState],
        obs_low: Mapping[int, AgentObsLow],
    ) -> tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        agent_ids, agent_states, goal_states, lidar_states = self._build_batch_inputs(states, obs_low)
        u_refs = self._u_ref_batch(goal_states, agent_states)
        h_all, _is_obs_all, lf_all, lg_all, responsibility_all = self._pairwise_terms_batch(agent_states, lidar_states)

        actions: Dict[int, np.ndarray] = {}
        relax: Dict[int, np.ndarray] = {}
        for idx, aid in enumerate(agent_ids):
            action_i, relax_i = self._solve_single(
                u_refs[idx],
                h_all[idx],
                lf_all[idx],
                lg_all[idx],
                responsibility_all[idx],
            )
            actions[aid] = action_i.astype(np.float32)
            relax[aid] = relax_i.astype(np.float32)
        return actions, relax
