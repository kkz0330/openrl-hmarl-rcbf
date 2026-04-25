from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - optional backend
    cp = None  # type: ignore[assignment]

from hmarl_cbf.control import ConstraintBuilder, DifferentiableQPSolver
from hmarl_cbf.env.obstacles import copy_obstacle, extract_lidar_cbf_obstacles
from hmarl_cbf.types import AgentObsLow, AgentState, QPParam, QPSolution


@dataclass(slots=True)
class DistributedCBFBaselineConfig:
    nominal_mode: str = "lqr"
    use_clf: bool = True
    cbf_mode: str = "distributed_hocbf54"
    cbf_share_agent: float = 0.5
    cbf_share_obs: float = 1.0
    cbf_eps: float = 1e-4
    cbf_u_max: float = 2.0
    robust_cbf: bool = False
    disturbance_accel_max: float = 0.0
    relative_disturbance_accel_max: float = 0.0
    cbf_k0: float = 1.0
    cbf_k1: float = 1.0
    hocbf_gamma_h: float = 1.0
    hocbf_gamma_hdot: float = 1.0
    clf_k: float = 1.0
    H_diag: Tuple[float, float] = (1.0, 1.0)
    w_clf: float = 10.0
    w_cbf: float = 100.0
    cbf_slack_max: float = 0.5
    ref_speed: float = 1.2
    speed_kp: float = 1.2
    slow_radius: float = 1.5
    goal_stop_min_speed: float = 0.0
    lqr_q_pos: float = 5.0
    lqr_q_vel: float = 5.0
    lqr_r_input: float = 1.0
    lqr_error_clip_radius: float = 3.0
    gcbf_mass: float = 1.0
    gcbf_comm_radius: float = 3.0
    dt: float = 0.03
    velocity_limit: float = 2.0
    mpc_horizon: int = 12
    mpc_q_pos: float = 6.0
    mpc_q_vel: float = 0.8
    mpc_q_terminal_pos: float = 10.0
    mpc_q_terminal_vel: float = 1.0
    mpc_r_input: float = 0.15
    mpc_obs_extra_margin: float = 0.0
    mpc_obs_constraint_horizon: int = 6
    neighbor_radius: float = 2.0
    obstacle_range: float = 3.0
    prefilter_neighbors: bool = True
    prefilter_obstacles_top_k: bool = True
    boundary_cbf: bool = True
    boundary_margin: float = 0.3
    world_size: float = 10.0
    rect_base_margin_extra: float = 0.0
    rect_corner_margin_enabled: bool = False
    rect_corner_margin_max: float = 0.0
    rect_corner_proximity_distance: float = 0.4
    rect_corner_speed_min: float = 0.05
    rect_corner_alignment_power: float = 1.0
    rect_dual_edge_cbf_enabled: bool = False
    rect_dual_edge_proximity_distance: float = 0.0
    rect_smooth_tau: float = 0.1
    lidar_cbf_use_fitted_geometry: bool = False
    lidar_cbf_point_radius: float = 0.0
    lidar_cbf_top_k: int = 3
    lidar_cbf_min_segment_points: int = 2
    lidar_cbf_line_fit_max_residual: float = 0.08
    lidar_cbf_circle_fit_max_residual: float = 0.08
    lidar_cbf_circle_radius_min: float = 0.05
    lidar_cbf_circle_radius_max: float = 100.0
    use_input_bounds: bool = True
    unbounded_action_limit: float = 1e6

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "DistributedCBFBaselineConfig":
        h_raw = data.get("H_diag", data.get("r_diag", (1.0, 1.0)))
        h_arr = np.asarray(h_raw, dtype=np.float32).reshape(-1)
        if h_arr.shape[0] != 2:
            raise ValueError(f"baseline.H_diag must contain 2 values, got {h_arr.shape[0]}")
        return DistributedCBFBaselineConfig(
            nominal_mode=str(data.get("nominal_mode", "lqr")),
            use_clf=bool(data.get("use_clf", True)),
            cbf_mode=str(data.get("cbf_mode", "distributed_hocbf54")),
            cbf_share_agent=float(data.get("cbf_share_agent", 0.5)),
            cbf_share_obs=float(data.get("cbf_share_obs", 1.0)),
            cbf_eps=float(data.get("cbf_eps", 1e-4)),
            cbf_u_max=float(data.get("cbf_u_max", 2.0)),
            robust_cbf=bool(data.get("robust_cbf", False)),
            disturbance_accel_max=float(data.get("disturbance_accel_max", 0.0)),
            relative_disturbance_accel_max=float(data.get("relative_disturbance_accel_max", 0.0)),
            cbf_k0=float(data.get("cbf_k0", 1.0)),
            cbf_k1=float(data.get("cbf_k1", 1.0)),
            hocbf_gamma_h=float(data.get("hocbf_gamma_h", 1.0)),
            hocbf_gamma_hdot=float(data.get("hocbf_gamma_hdot", 1.0)),
            clf_k=float(data.get("clf_k", 1.0)),
            H_diag=(float(h_arr[0]), float(h_arr[1])),
            w_clf=float(data.get("w_clf", 10.0)),
            w_cbf=float(data.get("w_cbf", 100.0)),
            cbf_slack_max=float(data.get("cbf_slack_max", 0.5)),
            ref_speed=float(data.get("ref_speed", 1.2)),
            speed_kp=float(data.get("speed_kp", 1.2)),
            slow_radius=float(data.get("slow_radius", 1.5)),
            goal_stop_min_speed=float(data.get("goal_stop_min_speed", 0.0)),
            lqr_q_pos=float(data.get("lqr_q_pos", 5.0)),
            lqr_q_vel=float(data.get("lqr_q_vel", 5.0)),
            lqr_r_input=float(data.get("lqr_r_input", 1.0)),
            lqr_error_clip_radius=float(data.get("lqr_error_clip_radius", data.get("obstacle_range", 3.0))),
            gcbf_mass=float(data.get("gcbf_mass", 1.0)),
            gcbf_comm_radius=float(data.get("gcbf_comm_radius", data.get("neighbor_radius", 3.0))),
            dt=float(data.get("dt", 0.03)),
            velocity_limit=float(data.get("velocity_limit", 2.0)),
            mpc_horizon=int(data.get("mpc_horizon", 12)),
            mpc_q_pos=float(data.get("mpc_q_pos", 6.0)),
            mpc_q_vel=float(data.get("mpc_q_vel", 0.8)),
            mpc_q_terminal_pos=float(data.get("mpc_q_terminal_pos", 10.0)),
            mpc_q_terminal_vel=float(data.get("mpc_q_terminal_vel", 1.0)),
            mpc_r_input=float(data.get("mpc_r_input", 0.15)),
            mpc_obs_extra_margin=float(data.get("mpc_obs_extra_margin", 0.0)),
            mpc_obs_constraint_horizon=int(data.get("mpc_obs_constraint_horizon", 6)),
            neighbor_radius=float(data.get("neighbor_radius", 2.0)),
            obstacle_range=float(data.get("obstacle_range", 3.0)),
            prefilter_neighbors=bool(data.get("prefilter_neighbors", True)),
            prefilter_obstacles_top_k=bool(data.get("prefilter_obstacles_top_k", True)),
            boundary_cbf=bool(data.get("boundary_cbf", True)),
            boundary_margin=float(data.get("boundary_margin", 0.3)),
            world_size=float(data.get("world_size", 10.0)),
            rect_base_margin_extra=float(data.get("rect_base_margin_extra", 0.0)),
            rect_corner_margin_enabled=bool(data.get("rect_corner_margin_enabled", False)),
            rect_corner_margin_max=float(data.get("rect_corner_margin_max", 0.0)),
            rect_corner_proximity_distance=float(data.get("rect_corner_proximity_distance", 0.4)),
            rect_corner_speed_min=float(data.get("rect_corner_speed_min", 0.05)),
            rect_corner_alignment_power=float(data.get("rect_corner_alignment_power", 1.0)),
            rect_dual_edge_cbf_enabled=bool(data.get("rect_dual_edge_cbf_enabled", False)),
            rect_dual_edge_proximity_distance=float(data.get("rect_dual_edge_proximity_distance", 0.0)),
            rect_smooth_tau=float(data.get("rect_smooth_tau", 0.1)),
            lidar_cbf_use_fitted_geometry=bool(data.get("lidar_cbf_use_fitted_geometry", False)),
            lidar_cbf_point_radius=float(data.get("lidar_cbf_point_radius", 0.0)),
            lidar_cbf_top_k=int(data.get("lidar_cbf_top_k", 3)),
            lidar_cbf_min_segment_points=int(data.get("lidar_cbf_min_segment_points", 2)),
            lidar_cbf_line_fit_max_residual=float(data.get("lidar_cbf_line_fit_max_residual", 0.08)),
            lidar_cbf_circle_fit_max_residual=float(data.get("lidar_cbf_circle_fit_max_residual", 0.08)),
            lidar_cbf_circle_radius_min=float(data.get("lidar_cbf_circle_radius_min", 0.05)),
            lidar_cbf_circle_radius_max=float(data.get("lidar_cbf_circle_radius_max", 100.0)),
            use_input_bounds=bool(data.get("use_input_bounds", True)),
            unbounded_action_limit=float(data.get("unbounded_action_limit", 1e6)),
        )


class DistributedCBFBaselineController:
    """
    Fixed-parameter distributed CBF-QP baseline controller.

    This controller mirrors the hand-crafted distributed CBF-QP baseline style:
    - Goal-directed nominal control encoded as QP linear term F
    - Soft CBF constraints (agent-agent + agent-obstacle)
    - Soft CLF and optional input bounds
    """

    def __init__(
        self,
        constraint_builder: ConstraintBuilder,
        qp_solver: DifferentiableQPSolver,
        config: DistributedCBFBaselineConfig,
    ) -> None:
        self.constraint_builder = constraint_builder
        self.qp_solver = qp_solver
        self.config = config
        self._mpc_problem_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._lqr_gain_cache: np.ndarray | None = None
        self._gcbf_lqr_gain_cache: np.ndarray | None = None

    @staticmethod
    def _unit(vec: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(vec))
        if n <= 1e-8:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return (vec / n).astype(np.float32)

    def _goal_tracking_f_lin(
        self,
        state: AgentState,
        obstacles_local: List[Dict[str, np.ndarray | float]] | None = None,
    ) -> np.ndarray:
        mode = str(self.config.nominal_mode).strip().lower()
        if mode == "gcbf_lqr":
            return self._gcbf_lqr_goal_tracking_f_lin(state)
        if mode == "mpc":
            return self._mpc_goal_tracking_f_lin(state, list(obstacles_local or []))
        if mode == "pd":
            return self._pd_goal_tracking_f_lin(state)
        return self._lqr_goal_tracking_f_lin(state)

    def _pd_goal_tracking_f_lin(self, state: AgentState) -> np.ndarray:
        goal_vec = np.asarray(state.goal - state.position, dtype=np.float32).reshape(2)
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist > 1e-8:
            goal_dir = goal_vec / goal_dist
        else:
            vel = np.asarray(state.velocity, dtype=np.float32).reshape(2)
            v_norm = float(np.linalg.norm(vel))
            if v_norm > 1e-8:
                goal_dir = vel / v_norm
            else:
                goal_dir = np.asarray([1.0, 0.0], dtype=np.float32)

        speed_des = float(self.config.ref_speed)
        if self.config.slow_radius > 0.0:
            speed_scale = float(np.clip(goal_dist / max(self.config.slow_radius, 1e-8), 0.0, 1.0))
            speed_des = max(float(self.config.goal_stop_min_speed), speed_des * speed_scale)

        v_des = speed_des * goal_dir
        v = np.asarray(state.velocity, dtype=np.float32).reshape(2)
        a_des = (float(self.config.speed_kp) * (v_des - v)).astype(np.float32)
        H_mat = np.diag(np.asarray(self.config.H_diag, dtype=np.float32).reshape(2)).astype(np.float32)
        return -(H_mat @ a_des).astype(np.float32)

    def _desired_velocity(self, state: AgentState) -> np.ndarray:
        goal_vec = np.asarray(state.goal - state.position, dtype=np.float32).reshape(2)
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist > 1e-8:
            goal_dir = goal_vec / goal_dist
        else:
            vel = np.asarray(state.velocity, dtype=np.float32).reshape(2)
            v_norm = float(np.linalg.norm(vel))
            if v_norm > 1e-8:
                goal_dir = vel / v_norm
            else:
                goal_dir = np.asarray([1.0, 0.0], dtype=np.float32)

        speed_des = float(self.config.ref_speed)
        if self.config.slow_radius > 0.0:
            speed_scale = float(np.clip(goal_dist / max(self.config.slow_radius, 1e-8), 0.0, 1.0))
            speed_des = max(float(self.config.goal_stop_min_speed), speed_des * speed_scale)
        return (speed_des * goal_dir).astype(np.float32)

    def _get_lqr_gain(self) -> np.ndarray:
        if self._lqr_gain_cache is not None:
            return self._lqr_gain_cache

        dt = float(max(1e-4, self.config.dt))
        A = np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        B = np.asarray(
            [
                [dt * dt, 0.0],
                [0.0, dt * dt],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=np.float32,
        )
        Q = np.diag(
            np.asarray(
                [
                    float(self.config.lqr_q_pos),
                    float(self.config.lqr_q_pos),
                    float(self.config.lqr_q_vel),
                    float(self.config.lqr_q_vel),
                ],
                dtype=np.float32,
            )
        ).astype(np.float32)
        R = np.diag(
            np.asarray(
                [float(self.config.lqr_r_input), float(self.config.lqr_r_input)],
                dtype=np.float32,
            )
        ).astype(np.float32)

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

    def _lqr_goal_tracking_f_lin(self, state: AgentState) -> np.ndarray:
        K = self._get_lqr_gain()
        pos_err = np.asarray(state.goal - state.position, dtype=np.float32).reshape(2)
        vel = np.asarray(state.velocity, dtype=np.float32).reshape(2)
        error = np.concatenate([pos_err, -vel], axis=0).astype(np.float32)
        error_norm = float(np.linalg.norm(error))
        clip_radius = float(max(1e-6, self.config.lqr_error_clip_radius))
        if error_norm > 1e-8:
            error_max = np.abs(error / error_norm * clip_radius).astype(np.float32)
            error = np.clip(error, -error_max, error_max).astype(np.float32)
        u_ref = (error @ K.T).astype(np.float32)
        u_limit = float(max(1e-6, self.config.cbf_u_max))
        u_ref = np.clip(u_ref, -u_limit, u_limit).astype(np.float32)
        H_mat = np.diag(np.asarray(self.config.H_diag, dtype=np.float32).reshape(2)).astype(np.float32)
        return -(H_mat @ u_ref).astype(np.float32)

    def _get_gcbf_lqr_gain(self) -> np.ndarray:
        if self._gcbf_lqr_gain_cache is not None:
            return self._gcbf_lqr_gain_cache

        dt = float(max(1e-6, self.config.dt))
        inv_m = 1.0 / float(max(1e-6, self.config.gcbf_mass))
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
                [
                    float(self.config.lqr_q_pos),
                    float(self.config.lqr_q_pos),
                    float(self.config.lqr_q_vel),
                    float(self.config.lqr_q_vel),
                ],
                dtype=np.float32,
            )
        ).astype(np.float32)
        R = np.diag(
            np.asarray(
                [float(self.config.lqr_r_input), float(self.config.lqr_r_input)],
                dtype=np.float32,
            )
        ).astype(np.float32)

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
        self._gcbf_lqr_gain_cache = gain.astype(np.float32)
        return self._gcbf_lqr_gain_cache

    def _gcbf_lqr_goal_tracking_f_lin(self, state: AgentState) -> np.ndarray:
        K = self._get_gcbf_lqr_gain()
        agent_state = np.asarray(
            [state.position[0], state.position[1], state.velocity[0], state.velocity[1]],
            dtype=np.float32,
        )
        goal_state = np.asarray([state.goal[0], state.goal[1], 0.0, 0.0], dtype=np.float32)
        error = (goal_state - agent_state).astype(np.float32)
        error_norm = float(np.linalg.norm(error))
        clip_radius = float(max(1e-6, self.config.gcbf_comm_radius))
        if error_norm > 1e-8:
            error_max = np.abs(error / error_norm * clip_radius).astype(np.float32)
            error = np.clip(error, -error_max, error_max).astype(np.float32)
        u_ref = (error @ K.T).astype(np.float32)
        u_limit = float(max(1e-6, self.config.cbf_u_max))
        u_ref = np.clip(u_ref, -u_limit, u_limit).astype(np.float32)
        return (-u_ref).astype(np.float32)

    def _get_mpc_problem(self, n_obs: int) -> Dict[str, Any] | None:
        if cp is None:
            return None
        horizon = int(max(1, self.config.mpc_horizon))
        n_obs = int(max(0, n_obs))
        cache_key = (horizon, n_obs)
        if cache_key in self._mpc_problem_cache:
            return self._mpc_problem_cache[cache_key]

        dt = float(max(1e-4, self.config.dt))
        A = np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        B = np.asarray(
            [
                [dt * dt, 0.0],
                [0.0, dt * dt],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=np.float32,
        )

        x0 = cp.Parameter(4)
        x_ref = cp.Parameter(4)
        u_bound = cp.Parameter(nonneg=True)
        v_bound = cp.Parameter(nonneg=True)
        if n_obs > 0:
            obs_normals = cp.Parameter((n_obs, 2))
            obs_offsets = cp.Parameter(n_obs)
        else:
            obs_normals = None
            obs_offsets = None
        X = cp.Variable((4, horizon + 1))
        U = cp.Variable((2, horizon))

        q_pos = float(self.config.mpc_q_pos)
        q_vel = float(self.config.mpc_q_vel)
        q_term_pos = float(self.config.mpc_q_terminal_pos)
        q_term_vel = float(self.config.mpc_q_terminal_vel)
        r_input = float(self.config.mpc_r_input)

        cost = 0
        constraints = [X[:, 0] == x0]
        obs_horizon = int(max(1, min(horizon, self.config.mpc_obs_constraint_horizon)))
        for k in range(horizon):
            pos_err = X[:2, k] - x_ref[:2]
            vel_err = X[2:, k] - x_ref[2:]
            cost += q_pos * cp.sum_squares(pos_err)
            cost += q_vel * cp.sum_squares(vel_err)
            cost += r_input * cp.sum_squares(U[:, k])
            constraints.append(X[:, k + 1] == A @ X[:, k] + B @ U[:, k])
            constraints.append(U[:, k] <= u_bound)
            constraints.append(U[:, k] >= -u_bound)
            constraints.append(X[2:, k + 1] <= v_bound)
            constraints.append(X[2:, k + 1] >= -v_bound)
            if n_obs > 0 and k < obs_horizon and obs_normals is not None and obs_offsets is not None:
                constraints.append(obs_normals @ X[:2, k + 1] >= obs_offsets)
        cost += q_term_pos * cp.sum_squares(X[:2, horizon] - x_ref[:2])
        cost += q_term_vel * cp.sum_squares(X[2:, horizon] - x_ref[2:])

        problem = cp.Problem(cp.Minimize(cost), constraints)
        cache = {
            "problem": problem,
            "x0": x0,
            "x_ref": x_ref,
            "u_bound": u_bound,
            "v_bound": v_bound,
            "obs_normals": obs_normals,
            "obs_offsets": obs_offsets,
            "U": U,
        }
        self._mpc_problem_cache[cache_key] = cache
        return cache

    def _linearized_obstacle_halfspaces(
        self,
        state: AgentState,
        obstacles_local: List[Dict[str, np.ndarray | float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(obstacles_local) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        p0 = np.asarray(state.position, dtype=np.float32).reshape(2)
        v0 = np.asarray(state.velocity, dtype=np.float32).reshape(2)
        goal_dir = self._unit(np.asarray(state.goal - state.position, dtype=np.float32).reshape(2))
        normals: List[np.ndarray] = []
        offsets: List[float] = []
        for obs in obstacles_local:
            center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
            radius = float(max(0.0, obs.get("radius", 0.0)))
            rel = p0 - center
            dist = float(np.linalg.norm(rel))
            if dist > 1e-6:
                normal = (rel / dist).astype(np.float32)
            else:
                if float(np.linalg.norm(v0)) > 1e-6:
                    normal = self._unit(v0)
                elif float(np.linalg.norm(goal_dir)) > 1e-6:
                    normal = -goal_dir
                else:
                    normal = np.asarray([1.0, 0.0], dtype=np.float32)
            safe_distance = float(self.constraint_builder.d_safe_obs + radius + self.config.mpc_obs_extra_margin)
            normals.append(normal.astype(np.float32))
            offsets.append(float(np.dot(normal, center) + safe_distance))
        return np.stack(normals, axis=0).astype(np.float32), np.asarray(offsets, dtype=np.float32)

    def _mpc_nominal_action(
        self,
        state: AgentState,
        obstacles_local: List[Dict[str, np.ndarray | float]],
    ) -> np.ndarray:
        obs_normals_np, obs_offsets_np = self._linearized_obstacle_halfspaces(state, obstacles_local)
        cache = self._get_mpc_problem(int(obs_normals_np.shape[0]))
        if cache is None:
            v_des = self._desired_velocity(state)
            v = np.asarray(state.velocity, dtype=np.float32).reshape(2)
            return (float(self.config.speed_kp) * (v_des - v)).astype(np.float32)
        x0 = np.asarray(
            [
                float(state.position[0]),
                float(state.position[1]),
                float(state.velocity[0]),
                float(state.velocity[1]),
            ],
            dtype=np.float32,
        )
        v_des = self._desired_velocity(state)
        x_ref = np.asarray(
            [
                float(state.goal[0]),
                float(state.goal[1]),
                float(v_des[0]),
                float(v_des[1]),
            ],
            dtype=np.float32,
        )
        cache["x0"].value = x0
        cache["x_ref"].value = x_ref
        cache["u_bound"].value = float(self.config.cbf_u_max)
        cache["v_bound"].value = float(self.config.velocity_limit)
        if cache.get("obs_normals", None) is not None:
            cache["obs_normals"].value = obs_normals_np
            cache["obs_offsets"].value = obs_offsets_np

        for solver_name, kwargs in (("OSQP", {"warm_start": True}), ("ECOS", {"warm_start": True}), ("SCS", {"warm_start": True})):
            try:
                cache["problem"].solve(solver=solver_name, **kwargs)
            except Exception:
                continue
            if cache["problem"].status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and cache["U"].value is not None:
                return np.asarray(cache["U"].value[:, 0], dtype=np.float32).reshape(2)
        return float(self.config.speed_kp) * (v_des - np.asarray(state.velocity, dtype=np.float32).reshape(2))

    def _mpc_goal_tracking_f_lin(
        self,
        state: AgentState,
        obstacles_local: List[Dict[str, np.ndarray | float]],
    ) -> np.ndarray:
        u_nom = np.asarray(self._mpc_nominal_action(state, obstacles_local), dtype=np.float32).reshape(2)
        H_mat = np.diag(np.asarray(self.config.H_diag, dtype=np.float32).reshape(2)).astype(np.float32)
        return -(H_mat @ u_nom).astype(np.float32)

    def _filter_neighbors(self, state_i: AgentState, neighbors_all: List[AgentState]) -> List[AgentState]:
        if not bool(self.config.prefilter_neighbors):
            return list(neighbors_all)
        radius = float(self.config.neighbor_radius)
        if radius <= 0.0:
            return []
        return [
            s
            for s in neighbors_all
            if float(np.linalg.norm(state_i.position - s.position)) <= radius
        ]

    def _filter_obstacles(
        self,
        state_i: AgentState,
        obstacles_all: List[Dict[str, np.ndarray | float]],
        obs_low_i: AgentObsLow | None,
    ) -> List[Dict[str, np.ndarray | float]]:
        max_range = float(self.config.obstacle_range)
        if max_range <= 0.0:
            return []
        if obs_low_i is None:
            raise ValueError("obs_low is required for LiDAR-based baseline obstacle fitting")
        top_k = int(self.config.lidar_cbf_top_k) if int(self.config.lidar_cbf_top_k) > 0 else None
        if not bool(self.config.prefilter_obstacles_top_k):
            top_k = None
        local = extract_lidar_cbf_obstacles(
            scan=obs_low_i.lidar_scan,
            max_range=max_range,
            use_fitted_geometry=bool(self.config.lidar_cbf_use_fitted_geometry),
            point_radius=float(self.config.lidar_cbf_point_radius),
            top_k=top_k,
            min_segment_points=int(self.config.lidar_cbf_min_segment_points),
            line_fit_max_residual=float(self.config.lidar_cbf_line_fit_max_residual),
            circle_fit_max_residual=float(self.config.lidar_cbf_circle_fit_max_residual),
            circle_radius_min=float(self.config.lidar_cbf_circle_radius_min),
            circle_radius_max=float(self.config.lidar_cbf_circle_radius_max),
        )
        return [copy_obstacle(obs) for obs in local]

    def _build_qp_param(
        self,
        state: AgentState,
        obstacles_local: List[Dict[str, np.ndarray | float]] | None = None,
    ) -> QPParam:
        mode = str(self.config.nominal_mode).strip().lower()
        if mode == "gcbf_lqr":
            H_mat = np.eye(2, dtype=np.float32)
        else:
            H_mat = np.diag(np.asarray(self.config.H_diag, dtype=np.float32).reshape(2)).astype(np.float32)
        return QPParam(
            H_mat=H_mat,
            f_lin=self._goal_tracking_f_lin(state, obstacles_local=obstacles_local),
            w_clf=np.asarray([0.0 if not self.config.use_clf else float(self.config.w_clf)], dtype=np.float32),
            w_cbf=np.asarray([float(self.config.w_cbf)], dtype=np.float32),
            cbf_slack_max=np.asarray([float(self.config.cbf_slack_max)], dtype=np.float32),
            cbf_k0=np.asarray([float(self.config.cbf_k0)], dtype=np.float32),
            cbf_k1=np.asarray([float(self.config.cbf_k1)], dtype=np.float32),
            clf_k=np.asarray([float(self.config.clf_k)], dtype=np.float32),
            hocbf_gamma_h=np.asarray([float(self.config.hocbf_gamma_h)], dtype=np.float32),
            hocbf_gamma_hdot=np.asarray([float(self.config.hocbf_gamma_hdot)], dtype=np.float32),
        )

    def solve_batch(
        self,
        states: Mapping[int, AgentState],
        obstacles: List[Dict[str, np.ndarray | float]],
        obs_low: Mapping[int, AgentObsLow] | None = None,
    ) -> tuple[Dict[int, np.ndarray], Dict[int, QPSolution]]:
        actions: Dict[int, np.ndarray] = {}
        solutions: Dict[int, QPSolution] = {}
        for agent_id, state_i in states.items():
            neighbors_all = [s for other_id, s in states.items() if other_id != agent_id]
            neighbors = self._filter_neighbors(state_i=state_i, neighbors_all=neighbors_all)
            obs_low_i = None if obs_low is None else obs_low.get(agent_id, None)
            obstacles_local = self._filter_obstacles(state_i=state_i, obstacles_all=obstacles, obs_low_i=obs_low_i)
            qp_param = self._build_qp_param(state_i, obstacles_local=obstacles_local)
            overrides = {
                "cbf_mode": str(self.config.cbf_mode),
                "cbf_u_max": float(self.config.cbf_u_max),
                "robust_cbf": bool(self.config.robust_cbf),
                "disturbance_accel_max": float(self.config.disturbance_accel_max),
                "relative_disturbance_accel_max": float(self.config.relative_disturbance_accel_max),
                "cbf_share_agent": float(self.config.cbf_share_agent),
                "cbf_share_obs": float(self.config.cbf_share_obs),
                "cbf_eps": float(self.config.cbf_eps),
                "use_input_bounds": bool(self.config.use_input_bounds),
                "unbounded_action_limit": float(self.config.unbounded_action_limit),
                "target_speed": float(self.config.ref_speed),
                "slow_radius": float(self.config.slow_radius),
                "goal_stop_min_speed": float(self.config.goal_stop_min_speed),
                "boundary_cbf": bool(self.config.boundary_cbf),
                "boundary_margin": float(self.config.boundary_margin),
                "world_size": float(self.config.world_size),
                "rect_base_margin_extra": float(self.config.rect_base_margin_extra),
                "rect_corner_margin_enabled": bool(self.config.rect_corner_margin_enabled),
                "rect_corner_margin_max": float(self.config.rect_corner_margin_max),
                "rect_corner_proximity_distance": float(self.config.rect_corner_proximity_distance),
                "rect_corner_speed_min": float(self.config.rect_corner_speed_min),
                "rect_corner_alignment_power": float(self.config.rect_corner_alignment_power),
                "rect_dual_edge_cbf_enabled": bool(self.config.rect_dual_edge_cbf_enabled),
                "rect_dual_edge_proximity_distance": float(self.config.rect_dual_edge_proximity_distance),
                "rect_smooth_tau": float(self.config.rect_smooth_tau),
                "lidar_cbf_top_k": int(self.config.lidar_cbf_top_k),
                "use_clf": bool(self.config.use_clf),
                "w_cbf": float(self.config.w_cbf),
                "cbf_slack_max": float(self.config.cbf_slack_max),
            }
            problem = self.constraint_builder.build_for_agent(
                state_i=state_i,
                neighbors=neighbors,
                obstacles=obstacles_local,
                qp_param=qp_param,
                constraint_overrides=overrides,
            )
            solution = self.qp_solver.solve(problem)
            actions[agent_id] = np.asarray(solution.action, dtype=np.float32).reshape(2)
            solutions[agent_id] = solution
        return actions, solutions
