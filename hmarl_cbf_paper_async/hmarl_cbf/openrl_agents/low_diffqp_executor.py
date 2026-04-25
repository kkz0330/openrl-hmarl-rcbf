from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - import-safe fallback
    torch = None  # type: ignore[assignment]

from hmarl_cbf.control import ConstraintBuilder, TorchDifferentiableQPSolver, build_diff_constraint_constants
from hmarl_cbf.env.obstacles import copy_obstacle, extract_lidar_cbf_obstacles
from hmarl_cbf.openrl_models.low_qp_decoder import LowQPDecoder
from hmarl_cbf.types import AgentObsLow, AgentState, QPParam


@dataclass(slots=True)
class TorchDiffQPSolveOutput:
    agent_id: int
    skill_id: int
    phi: np.ndarray
    qp_param: QPParam
    action: np.ndarray
    slack: np.ndarray
    cbf_slack: np.ndarray
    b_cbf: np.ndarray
    b_clf: np.ndarray
    qp_feasible: bool
    qp_solver_status: str
    qp_used_fallback: bool
    neighbors_used: List[AgentState]
    obstacles_used: List[Dict[str, Any]]
    safety_constraints: Dict[str, Any]


class TorchDiffQPActionAdapter:
    """Low-level ``phi -> QPParam -> torch diff-QP -> safe action`` execution chain."""

    def __init__(
        self,
        *,
        decoder: LowQPDecoder,
        diff_qp_solver: TorchDifferentiableQPSolver,
        constraint_builder: ConstraintBuilder,
        neighbor_perception_radius: float | None = None,
        obstacle_perception_range: float | None = None,
        torch_device: str = "cpu",
    ) -> None:
        if torch is None:
            raise RuntimeError("TorchDiffQPActionAdapter requires PyTorch")
        self.decoder = decoder
        self.diff_qp_solver = diff_qp_solver
        self.constraint_builder = constraint_builder
        self.neighbor_perception_radius = (
            None if neighbor_perception_radius is None else float(max(0.0, neighbor_perception_radius))
        )
        self.obstacle_perception_range = (
            None if obstacle_perception_range is None else float(max(0.0, obstacle_perception_range))
        )
        self.device = torch.device(torch_device)
        self.decoder.to(self.device)
        self.decoder.eval()

    @staticmethod
    def _copy_state(state: AgentState) -> AgentState:
        return AgentState(
            agent_id=int(state.agent_id),
            position=np.asarray(state.position, dtype=np.float32).copy(),
            velocity=np.asarray(state.velocity, dtype=np.float32).copy(),
            goal=np.asarray(state.goal, dtype=np.float32).copy(),
            radius=float(state.radius),
        )

    @staticmethod
    def _scalar_from_any(value: Any) -> float:
        if value is None:
            raise ValueError("expected scalar-like value, got None")
        if torch is not None and isinstance(value, torch.Tensor):
            return float(value.detach().reshape(-1)[0].cpu().item())
        return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])

    @staticmethod
    def _flatten_qp_param_torch(qp_param_raw: QPParam) -> QPParam:
        return QPParam(
            H_mat=qp_param_raw.H_mat.reshape(2, 2),
            f_lin=qp_param_raw.f_lin.reshape(-1),
            w_clf=qp_param_raw.w_clf.reshape(-1),
            w_cbf=qp_param_raw.w_cbf.reshape(-1),
            cbf_slack_max=qp_param_raw.cbf_slack_max.reshape(-1),
            cbf_k0=qp_param_raw.cbf_k0.reshape(-1),
            cbf_k1=qp_param_raw.cbf_k1.reshape(-1),
            clf_k=qp_param_raw.clf_k.reshape(-1),
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
            d_min_agent=(
                qp_param_raw.d_min_agent.reshape(-1)
                if getattr(qp_param_raw, "d_min_agent", None) is not None
                else None
            ),
            d_safe_obs=(
                qp_param_raw.d_safe_obs.reshape(-1)
                if getattr(qp_param_raw, "d_safe_obs", None) is not None
                else None
            ),
        )

    @staticmethod
    def _qp_param_to_numpy(qp_param: QPParam) -> QPParam:
        def _to_np(value: Any) -> np.ndarray | None:
            if value is None:
                return None
            if torch is not None and isinstance(value, torch.Tensor):
                return np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
            return np.asarray(value, dtype=np.float32)

        return QPParam(
            H_mat=_to_np(qp_param.H_mat).reshape(2, 2),  # type: ignore[union-attr]
            f_lin=_to_np(qp_param.f_lin).reshape(-1),  # type: ignore[union-attr]
            w_clf=_to_np(qp_param.w_clf).reshape(-1),  # type: ignore[union-attr]
            w_cbf=_to_np(qp_param.w_cbf).reshape(-1),  # type: ignore[union-attr]
            cbf_slack_max=_to_np(qp_param.cbf_slack_max).reshape(-1),  # type: ignore[union-attr]
            cbf_k0=_to_np(qp_param.cbf_k0).reshape(-1),  # type: ignore[union-attr]
            cbf_k1=_to_np(qp_param.cbf_k1).reshape(-1),  # type: ignore[union-attr]
            clf_k=_to_np(qp_param.clf_k).reshape(-1),  # type: ignore[union-attr]
            hocbf_gamma_h=(
                _to_np(qp_param.hocbf_gamma_h).reshape(-1)  # type: ignore[union-attr]
                if getattr(qp_param, "hocbf_gamma_h", None) is not None
                else None
            ),
            hocbf_gamma_hdot=(
                _to_np(qp_param.hocbf_gamma_hdot).reshape(-1)  # type: ignore[union-attr]
                if getattr(qp_param, "hocbf_gamma_hdot", None) is not None
                else None
            ),
            d_min_agent=(
                _to_np(qp_param.d_min_agent).reshape(-1)  # type: ignore[union-attr]
                if getattr(qp_param, "d_min_agent", None) is not None
                else None
            ),
            d_safe_obs=(
                _to_np(qp_param.d_safe_obs).reshape(-1)  # type: ignore[union-attr]
                if getattr(qp_param, "d_safe_obs", None) is not None
                else None
            ),
        )

    def _filter_neighbors(self, state_i: AgentState, neighbors: List[AgentState]) -> List[AgentState]:
        if self.neighbor_perception_radius is None:
            return [self._copy_state(s) for s in neighbors]
        radius = float(self.neighbor_perception_radius)
        return [
            self._copy_state(state_j)
            for state_j in neighbors
            if float(np.linalg.norm(state_i.position - state_j.position)) <= radius
        ]

    def _filter_obstacles(self, obs_low: AgentObsLow) -> List[Dict[str, Any]]:
        scan = obs_low.lidar_scan
        max_range = (
            float(self.obstacle_perception_range)
            if self.obstacle_perception_range is not None
            else float(scan.max_range)
        )
        lidar_cfg = dict(getattr(self.constraint_builder, "lidar_cbf_config", {}) or {})
        perceived = extract_lidar_cbf_obstacles(
            scan=scan,
            max_range=max_range,
            use_fitted_geometry=bool(lidar_cfg.get("use_fitted_geometry", False)),
            point_radius=float(lidar_cfg.get("point_radius", 0.0)),
            top_k=int(lidar_cfg.get("top_k", 0)) if int(lidar_cfg.get("top_k", 0)) > 0 else None,
            min_segment_points=int(lidar_cfg.get("min_segment_points", 2)),
            line_fit_max_residual=float(lidar_cfg.get("line_fit_max_residual", 0.08)),
            circle_fit_max_residual=float(lidar_cfg.get("circle_fit_max_residual", 0.08)),
            circle_radius_min=float(lidar_cfg.get("circle_radius_min", 0.05)),
            circle_radius_max=float(lidar_cfg.get("circle_radius_max", 100.0)),
        )
        return [copy_obstacle(obs) for obs in perceived]

    def _build_diff_constants(
        self,
        *,
        state_i: AgentState,
        neighbors: List[AgentState],
        obstacles: List[Dict[str, Any]],
        safety_constraints: Mapping[str, Any],
        qp_param: QPParam | None = None,
    ):
        overrides = dict(safety_constraints or {})
        use_input_bounds = bool(overrides.get("use_input_bounds", True))
        if use_input_bounds:
            u_min = np.asarray(overrides.get("u_min", self.constraint_builder.u_min), dtype=np.float32).reshape(2)
            u_max = np.asarray(overrides.get("u_max", self.constraint_builder.u_max), dtype=np.float32).reshape(2)
        else:
            unbounded = float(overrides.get("unbounded_action_limit", 1.0e6))
            u_min = np.asarray([-unbounded, -unbounded], dtype=np.float32)
            u_max = np.asarray([unbounded, unbounded], dtype=np.float32)

        clf_v_des_vector = overrides.get("clf_v_des_vector")
        if clf_v_des_vector is not None:
            clf_v_des_vector = np.asarray(clf_v_des_vector, dtype=np.float32).reshape(2)

        qp_d_min_agent = (
            self._scalar_from_any(qp_param.d_min_agent)
            if qp_param is not None and getattr(qp_param, "d_min_agent", None) is not None
            else None
        )
        qp_d_safe_obs = (
            self._scalar_from_any(qp_param.d_safe_obs)
            if qp_param is not None and getattr(qp_param, "d_safe_obs", None) is not None
            else None
        )

        return build_diff_constraint_constants(
            state_i=state_i,
            neighbors=neighbors,
            obstacles=obstacles,
            d_min_agent=float(qp_d_min_agent if qp_d_min_agent is not None else overrides.get("d_min_agent", self.constraint_builder.d_min_agent)),
            d_safe_obs=float(qp_d_safe_obs if qp_d_safe_obs is not None else overrides.get("d_safe_obs", self.constraint_builder.d_safe_obs)),
            cbf_mode=str(overrides.get("cbf_mode", "distributed_ecbf")),
            cbf_u_max=float(overrides.get("cbf_u_max", max(np.max(np.abs(u_min)), np.max(np.abs(u_max)), 1e-3))),
            cbf_share_agent=float(overrides.get("cbf_share_agent", 0.5)),
            cbf_share_obs=float(overrides.get("cbf_share_obs", 1.0)),
            cbf_eps=float(overrides.get("cbf_eps", 1e-4)),
            boundary_cbf=bool(overrides.get("boundary_cbf", False)),
            world_size=float(overrides.get("world_size", 0.0)),
            boundary_margin=float(overrides.get("boundary_margin", 0.0)),
            rect_corner_margin_enabled=bool(overrides.get("rect_corner_margin_enabled", False)),
            rect_base_margin_extra=float(overrides.get("rect_base_margin_extra", 0.0)),
            rect_corner_margin_max=float(overrides.get("rect_corner_margin_max", 0.0)),
            rect_corner_proximity_distance=float(overrides.get("rect_corner_proximity_distance", 0.4)),
            rect_corner_speed_min=float(overrides.get("rect_corner_speed_min", 0.05)),
            rect_corner_alignment_power=float(overrides.get("rect_corner_alignment_power", 1.0)),
            rect_dual_edge_cbf_enabled=bool(overrides.get("rect_dual_edge_cbf_enabled", False)),
            rect_dual_edge_proximity_distance=float(overrides.get("rect_dual_edge_proximity_distance", 0.0)),
            rect_smooth_tau=float(overrides.get("rect_smooth_tau", 0.1)),
            lidar_cbf_top_k=int(overrides.get("lidar_cbf_top_k", self.constraint_builder.lidar_cbf_config.get("top_k", 0))),
            robust_cbf=bool(overrides.get("robust_cbf", False)),
            disturbance_accel_max=float(overrides.get("disturbance_accel_max", 0.0)),
            relative_disturbance_accel_max=float(overrides.get("relative_disturbance_accel_max", 0.0)),
            clf_v_des_speed=float(
                overrides.get(
                    "clf_v_des_speed",
                    overrides.get(
                        "target_speed",
                        overrides.get(
                            "cruise_ref_speed",
                            overrides.get(
                                "decelerate_target_speed",
                                overrides.get("ref_speed", 0.8),
                            ),
                        ),
                    ),
                )
            ),
            clf_v_des_vector=clf_v_des_vector,
            u_min=u_min,
            u_max=u_max,
            device=str(self.device),
        )

    def solve_torch_for_agent(
        self,
        *,
        agent_id: int,
        state_i: AgentState,
        neighbors: List[AgentState],
        obs_low: AgentObsLow,
        skill_id: int,
        phi: np.ndarray | torch.Tensor,
        safety_constraints: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, QPParam, Any, List[AgentState], List[Dict[str, Any]]]:
        if torch is None:
            raise RuntimeError("TorchDiffQPActionAdapter requires PyTorch")

        neighbors_used = self._filter_neighbors(state_i, neighbors)
        obstacles_used = self._filter_obstacles(obs_low)
        obs_tensor = torch.as_tensor(obs_low.flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        phi_tensor = torch.as_tensor(phi, dtype=torch.float32, device=self.device).reshape(1, -1)
        qp_param_raw = self.decoder(obs_tensor, phi_tensor)
        qp_param = self._flatten_qp_param_torch(qp_param_raw)
        constants = self._build_diff_constants(
            state_i=state_i,
            neighbors=neighbors_used,
            obstacles=obstacles_used,
            safety_constraints=dict(safety_constraints or {}),
            qp_param=qp_param,
        )
        solve_out = self.diff_qp_solver.solve(qp_param, constants)
        return solve_out.action.reshape(2), qp_param, solve_out, neighbors_used, obstacles_used

    def solve_numpy_for_agent(
        self,
        *,
        agent_id: int,
        state_i: AgentState,
        neighbors: List[AgentState],
        obs_low: AgentObsLow,
        skill_id: int,
        phi: np.ndarray,
        safety_constraints: Mapping[str, Any] | None = None,
    ) -> TorchDiffQPSolveOutput:
        with torch.no_grad():
            action_t, qp_param_t, solve_out, neighbors_used, obstacles_used = self.solve_torch_for_agent(
                agent_id=agent_id,
                state_i=state_i,
                neighbors=neighbors,
                obs_low=obs_low,
                skill_id=skill_id,
                phi=phi,
                safety_constraints=safety_constraints,
            )
        return TorchDiffQPSolveOutput(
            agent_id=int(agent_id),
            skill_id=int(skill_id),
            phi=np.asarray(phi, dtype=np.float32).reshape(-1).copy(),
            qp_param=self._qp_param_to_numpy(qp_param_t),
            action=np.asarray(action_t.detach().cpu().numpy(), dtype=np.float32).reshape(2),
            slack=np.asarray(solve_out.slack.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            cbf_slack=np.asarray(solve_out.cbf_slack.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            b_cbf=np.asarray(solve_out.b_cbf.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            b_clf=np.asarray(solve_out.b_clf.detach().cpu().numpy(), dtype=np.float32).reshape(-1),
            qp_feasible=bool(solve_out.feasible and not solve_out.used_fallback),
            qp_solver_status=str(solve_out.solver_status),
            qp_used_fallback=bool(solve_out.used_fallback),
            neighbors_used=neighbors_used,
            obstacles_used=obstacles_used,
            safety_constraints=dict(safety_constraints or {}),
        )

    def __call__(
        self,
        states: Mapping[int, AgentState],
        obs_low: Mapping[int, AgentObsLow],
        skill_targets: Mapping[int, Dict[str, Any]],
        phi_actions: Mapping[int, np.ndarray],
        core_env,
    ) -> tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        del core_env
        actions: Dict[int, np.ndarray] = {}
        infos: Dict[int, Dict[str, Any]] = {}
        for agent_id, state_i in states.items():
            target = dict(skill_targets[int(agent_id)])
            neighbors_all = [state_j for other_id, state_j in states.items() if int(other_id) != int(agent_id)]
            solve_out = self.solve_numpy_for_agent(
                agent_id=int(agent_id),
                state_i=state_i,
                neighbors=neighbors_all,
                obs_low=obs_low[int(agent_id)],
                skill_id=int(target["skill_id"]),
                phi=np.asarray(phi_actions[int(agent_id)], dtype=np.float32).reshape(-1),
                safety_constraints=target.get("safety_constraints", {}),
            )
            actions[int(agent_id)] = solve_out.action.reshape(2).copy()
            infos[int(agent_id)] = {
                "adapter_mode": "torch_diff_qp",
                "phi": solve_out.phi.copy(),
                "executed_action": solve_out.action.copy(),
                "qp_feasible": bool(solve_out.qp_feasible),
                "qp_solver_status": str(solve_out.qp_solver_status),
                "qp_used_fallback": bool(solve_out.qp_used_fallback),
                "qp_H": np.asarray(solve_out.qp_param.H_mat, dtype=np.float32).reshape(2, 2).copy(),
                "qp_f": np.asarray(solve_out.qp_param.f_lin, dtype=np.float32).reshape(2).copy(),
                "qp_slack": solve_out.slack.copy(),
                "qp_cbf_slack": solve_out.cbf_slack.copy(),
                "qp_b_cbf": solve_out.b_cbf.copy(),
                "qp_b_clf": solve_out.b_clf.copy(),
                "neighbors_used": len(solve_out.neighbors_used),
                "obstacles_used": len(solve_out.obstacles_used),
                "perceived_neighbors": [
                    {
                        "agent_id": int(s.agent_id),
                        "position": np.asarray(s.position, dtype=np.float32).reshape(2).copy(),
                        "velocity": np.asarray(s.velocity, dtype=np.float32).reshape(2).copy(),
                        "goal": np.asarray(s.goal, dtype=np.float32).reshape(2).copy(),
                        "radius": float(s.radius),
                    }
                    for s in solve_out.neighbors_used
                ],
                "perceived_obstacles": [
                    copy_obstacle(dict(obs))
                    for obs in solve_out.obstacles_used
                ],
                "safety_constraints": dict(solve_out.safety_constraints),
            }
        return actions, infos
