from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from hmarl_cbf.control import ConstraintBuilder, DifferentiableQPSolver
from hmarl_cbf.types import AgentState, QPParam, QPSolution


@dataclass(slots=True)
class DistributedCBFBaselineConfig:
    cbf_mode: str = "distributed_hocbf54"
    cbf_share_agent: float = 0.5
    cbf_share_obs: float = 1.0
    cbf_eps: float = 1e-4
    cbf_u_max: float = 2.0
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
    neighbor_radius: float = 2.0
    obstacle_range: float = 3.0
    boundary_cbf: bool = True
    boundary_margin: float = 0.3
    world_size: float = 10.0
    use_input_bounds: bool = True
    unbounded_action_limit: float = 1e6

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "DistributedCBFBaselineConfig":
        h_raw = data.get("H_diag", data.get("r_diag", (1.0, 1.0)))
        h_arr = np.asarray(h_raw, dtype=np.float32).reshape(-1)
        if h_arr.shape[0] != 2:
            raise ValueError(f"baseline.H_diag must contain 2 values, got {h_arr.shape[0]}")
        return DistributedCBFBaselineConfig(
            cbf_mode=str(data.get("cbf_mode", "distributed_hocbf54")),
            cbf_share_agent=float(data.get("cbf_share_agent", 0.5)),
            cbf_share_obs=float(data.get("cbf_share_obs", 1.0)),
            cbf_eps=float(data.get("cbf_eps", 1e-4)),
            cbf_u_max=float(data.get("cbf_u_max", 2.0)),
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
            neighbor_radius=float(data.get("neighbor_radius", 2.0)),
            obstacle_range=float(data.get("obstacle_range", 3.0)),
            boundary_cbf=bool(data.get("boundary_cbf", True)),
            boundary_margin=float(data.get("boundary_margin", 0.3)),
            world_size=float(data.get("world_size", 10.0)),
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

    @staticmethod
    def _unit(vec: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(vec))
        if n <= 1e-8:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return (vec / n).astype(np.float32)

    def _goal_tracking_f_lin(self, state: AgentState) -> np.ndarray:
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

    def _filter_neighbors(self, state_i: AgentState, neighbors_all: List[AgentState]) -> List[AgentState]:
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
    ) -> List[Dict[str, np.ndarray | float]]:
        max_range = float(self.config.obstacle_range)
        if max_range <= 0.0:
            return []
        local: List[Dict[str, np.ndarray | float]] = []
        for obs in obstacles_all:
            center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
            radius = float(obs["radius"])
            surface_distance = float(np.linalg.norm(center - state_i.position) - radius)
            if surface_distance <= max_range:
                local.append({"center": center.copy(), "radius": radius})
        return local

    def _build_qp_param(self, state: AgentState) -> QPParam:
        H_mat = np.diag(np.asarray(self.config.H_diag, dtype=np.float32).reshape(2)).astype(np.float32)
        return QPParam(
            H_mat=H_mat,
            f_lin=self._goal_tracking_f_lin(state),
            w_clf=np.asarray([float(self.config.w_clf)], dtype=np.float32),
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
    ) -> tuple[Dict[int, np.ndarray], Dict[int, QPSolution]]:
        actions: Dict[int, np.ndarray] = {}
        solutions: Dict[int, QPSolution] = {}
        for agent_id, state_i in states.items():
            neighbors_all = [s for other_id, s in states.items() if other_id != agent_id]
            neighbors = self._filter_neighbors(state_i=state_i, neighbors_all=neighbors_all)
            obstacles_local = self._filter_obstacles(state_i=state_i, obstacles_all=obstacles)
            qp_param = self._build_qp_param(state_i)
            overrides = {
                "cbf_mode": str(self.config.cbf_mode),
                "cbf_u_max": float(self.config.cbf_u_max),
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
