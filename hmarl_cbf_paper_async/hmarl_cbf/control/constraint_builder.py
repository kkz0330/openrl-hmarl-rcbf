from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from hmarl_cbf.env.obstacles import normalize_obstacle, obstacle_barrier_geometry, obstacle_surface_distance
from hmarl_cbf.types import AgentState, QPParam, QPProblem


def _rect_base_extra_margin(overrides: Dict[str, Any]) -> float:
    return float(max(0.0, overrides.get("rect_base_margin_extra", 0.0)))


def _rect_smooth_tau(overrides: Dict[str, Any]) -> float:
    return float(max(1e-4, overrides.get("rect_smooth_tau", 0.1)))


def _robust_cbf_enabled(overrides: Dict[str, Any]) -> bool:
    return bool(overrides.get("robust_cbf", False))


def _disturbance_accel_max(overrides: Dict[str, Any]) -> float:
    return float(max(0.0, overrides.get("disturbance_accel_max", 0.0)))


def _relative_disturbance_accel_max(overrides: Dict[str, Any]) -> float:
    return float(max(0.0, overrides.get("relative_disturbance_accel_max", 0.0)))


def _robust_margin(a_row: np.ndarray, bound: float, enabled: bool) -> float:
    if (not enabled) or bound <= 0.0:
        return 0.0
    return float(np.linalg.norm(np.asarray(a_row, dtype=np.float32).reshape(-1)) * bound)


def _pointwise_top_k(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> int | None:
    raw = overrides.get("lidar_cbf_top_k", defaults.get("top_k", 0))
    top_k = int(max(0, raw))
    return top_k if top_k > 0 else None


def _collect_pointwise_candidates(
    state_i: AgentState,
    neighbors: List[AgentState],
    obstacles: List[Dict[str, np.ndarray | float]],
    *,
    d_min_agent: float,
    d_safe_obs: float,
    cbf_share_agent: float,
    cbf_share_obs: float,
    robust_cbf: bool,
    disturbance_accel_max: float,
    relative_disturbance_accel_max: float,
    top_k: int | None,
    rect_base_margin_extra: float,
    rect_smooth_tau: float,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for state_j in neighbors:
        p_rel = (state_i.position - state_j.position).astype(np.float32)
        v_rel = (state_i.velocity - state_j.velocity).astype(np.float32)
        candidates.append(
            {
                "kind": "agent",
                "a_row": (-2.0 * p_rel).astype(np.float32),
                "h0": float(np.dot(p_rel, p_rel) - max(d_min_agent, 1e-4) ** 2),
                "h0_dot": float(2.0 * np.dot(p_rel, v_rel)),
                "const_term": float(2.0 * np.dot(v_rel, v_rel)),
                "share": float(cbf_share_agent),
                "robust_bound": float(relative_disturbance_accel_max),
                "robust_enabled": bool(robust_cbf),
                "distance_key": float(np.linalg.norm(p_rel)),
            }
        )

    for obs in obstacles:
        obs_norm = normalize_obstacle(obs)
        vel = np.asarray(state_i.velocity, dtype=np.float32).reshape(2)
        geom = obstacle_barrier_geometry(
            state_i.position,
            obs_norm,
            inflation_margin=d_safe_obs + (rect_base_margin_extra if obs_norm["type"] == "rect" else 0.0),
            tau=rect_smooth_tau,
        )
        grad = np.asarray(geom["barrier_grad"], dtype=np.float32).reshape(2)
        hess = np.asarray(geom["barrier_hess"], dtype=np.float32).reshape(2, 2)
        candidates.append(
            {
                "kind": "obstacle",
                "a_row": (-grad).astype(np.float32),
                "h0": float(geom["barrier_h"]),
                "h0_dot": float(np.dot(grad, vel)),
                "const_term": float(vel @ hess @ vel),
                "share": float(cbf_share_obs),
                "robust_bound": float(disturbance_accel_max),
                "robust_enabled": bool(robust_cbf),
                "distance_key": float(obstacle_surface_distance(state_i.position, obs_norm)),
            }
        )

    if len(candidates) > 1:
        candidates.sort(key=lambda item: item["distance_key"])
    if top_k is not None and len(candidates) > top_k:
        candidates = candidates[:top_k]
    return candidates


class ConstraintBuilder:
    """Builds distributed hard-CBF and soft-CLF constraints for one agent."""

    def __init__(
        self,
        d_min_agent: float = 0.6,
        d_safe_obs: float = 0.6,
        u_min: np.ndarray | list[float] = (-1.0, -1.0),
        u_max: np.ndarray | list[float] = (1.0, 1.0),
        lidar_cbf_config: Dict[str, Any] | None = None,
    ) -> None:
        self.d_min_agent = float(d_min_agent)
        self.d_safe_obs = float(d_safe_obs)
        self.u_min = np.asarray(u_min, dtype=np.float32).reshape(2)
        self.u_max = np.asarray(u_max, dtype=np.float32).reshape(2)
        self.lidar_cbf_config = dict(lidar_cbf_config or {})

    def build_for_agent(
        self,
        state_i: AgentState,
        neighbors: List[AgentState],
        obstacles: List[Dict[str, np.ndarray | float]],
        qp_param: QPParam,
        H_override: np.ndarray | None = None,
        f_lin_override: np.ndarray | None = None,
        constraint_overrides: Dict[str, Any] | None = None,
    ) -> QPProblem:
        overrides = dict(constraint_overrides or {})
        qp_d_min_agent = getattr(qp_param, "d_min_agent", None)
        qp_d_safe_obs = getattr(qp_param, "d_safe_obs", None)
        if qp_d_min_agent is not None:
            d_min_agent = float(np.asarray(qp_d_min_agent).reshape(-1)[0])
        else:
            d_min_agent = float(overrides.get("d_min_agent", self.d_min_agent))
        if qp_d_safe_obs is not None:
            d_safe_obs = float(np.asarray(qp_d_safe_obs).reshape(-1)[0])
        else:
            d_safe_obs = float(overrides.get("d_safe_obs", self.d_safe_obs))
        use_input_bounds = bool(overrides.get("use_input_bounds", True))
        if use_input_bounds:
            u_min = np.asarray(overrides.get("u_min", self.u_min), dtype=np.float32).reshape(2)
            u_max = np.asarray(overrides.get("u_max", self.u_max), dtype=np.float32).reshape(2)
        else:
            unbounded = float(overrides.get("unbounded_action_limit", 1.0e6))
            u_min = np.asarray([-unbounded, -unbounded], dtype=np.float32)
            u_max = np.asarray([unbounded, unbounded], dtype=np.float32)
        cbf_mode = str(overrides.get("cbf_mode", "distributed_ecbf"))
        cbf_u_max = float(overrides.get("cbf_u_max", max(np.max(np.abs(u_min)), np.max(np.abs(u_max)), 1e-3)))
        cbf_share_agent = float(overrides.get("cbf_share_agent", 0.5))
        cbf_share_obs = float(overrides.get("cbf_share_obs", 1.0))
        cbf_eps = float(overrides.get("cbf_eps", 1e-4))
        pointwise_top_k = _pointwise_top_k(self.lidar_cbf_config, overrides)
        use_clf = bool(overrides.get("use_clf", True))
        boundary_cbf = bool(overrides.get("boundary_cbf", False))
        world_size = float(overrides.get("world_size", 0.0))
        boundary_margin = float(overrides.get("boundary_margin", 0.0))
        robust_cbf = _robust_cbf_enabled(overrides)
        disturbance_accel_max = _disturbance_accel_max(overrides)
        relative_disturbance_accel_max = _relative_disturbance_accel_max(overrides)

        A_cbf_rows: List[np.ndarray] = []
        b_cbf_rows: List[float] = []

        k0 = float(np.asarray(qp_param.cbf_k0).reshape(-1)[0])
        k1 = float(np.asarray(qp_param.cbf_k1).reshape(-1)[0])
        hocbf_gamma_h = (
            float(np.asarray(qp_param.hocbf_gamma_h).reshape(-1)[0])
            if qp_param.hocbf_gamma_h is not None
            else 1.0
        )
        hocbf_gamma_hdot = (
            float(np.asarray(qp_param.hocbf_gamma_hdot).reshape(-1)[0])
            if qp_param.hocbf_gamma_hdot is not None
            else 1.0
        )

        if cbf_mode == "distributed_gcbfplus":
            for candidate in _collect_pointwise_candidates(
                state_i,
                neighbors,
                obstacles,
                d_min_agent=d_min_agent,
                d_safe_obs=d_safe_obs,
                cbf_share_agent=cbf_share_agent,
                cbf_share_obs=cbf_share_obs,
                robust_cbf=robust_cbf,
                disturbance_accel_max=disturbance_accel_max,
                relative_disturbance_accel_max=relative_disturbance_accel_max,
                top_k=pointwise_top_k,
                rect_base_margin_extra=_rect_base_extra_margin(overrides),
                rect_smooth_tau=_rect_smooth_tau(overrides),
            ):
                a_row = np.asarray(candidate["a_row"], dtype=np.float32).reshape(2)
                h0 = float(candidate["h0"])
                h0_dot = float(candidate["h0_dot"])
                const_term = float(candidate["const_term"])
                alpha0 = k0 * hocbf_gamma_h
                alpha1 = k1 * hocbf_gamma_hdot
                h1 = h0_dot + alpha0 * h0
                lf_h1 = const_term + alpha0 * h0_dot
                b_row = float(candidate["share"]) * (lf_h1 + alpha1 * h1)
                b_row -= _robust_margin(a_row, float(candidate["robust_bound"]), bool(candidate["robust_enabled"]))
                A_cbf_rows.append(a_row)
                b_cbf_rows.append(float(b_row))
        else:
            # Agent-agent CBF (distributed local neighbors only).
            for state_j in neighbors:
                p_rel = state_i.position - state_j.position
                v_rel = state_i.velocity - state_j.velocity
                if cbf_mode == "distributed_hocbf54":
                    a_row, h_term, hdot_term, const_term = _build_hocbf54_row(
                        p_rel=p_rel,
                        v_rel=v_rel,
                        safe_distance=max(d_min_agent, 1e-4),
                        u_max=cbf_u_max,
                        share=cbf_share_agent,
                        eps=cbf_eps,
                    )
                else:
                    # ECBF-like linearization wrt u_i, distributed assumption on u_j.
                    h_term = float(np.dot(p_rel, p_rel) - d_min_agent**2)
                    hdot_term = float(2.0 * np.dot(p_rel, v_rel))
                    const_term = 2.0 * float(np.dot(v_rel, v_rel))
                    a_row = (-2.0 * p_rel).astype(np.float32)
                b_row = const_term + (k1 * hocbf_gamma_hdot) * hdot_term + (k0 * hocbf_gamma_h) * h_term
                b_row -= _robust_margin(a_row, relative_disturbance_accel_max, robust_cbf)
                A_cbf_rows.append(a_row)
                b_cbf_rows.append(float(b_row))

            # Agent-obstacle CBF (circle / point / rect obstacles).
            for obs in obstacles:
                obs_norm = normalize_obstacle(obs)
                vel = np.asarray(state_i.velocity, dtype=np.float32).reshape(2)
                geom = obstacle_barrier_geometry(
                    state_i.position,
                    obs_norm,
                    inflation_margin=d_safe_obs + (_rect_base_extra_margin(overrides) if obs_norm["type"] == "rect" else 0.0),
                    tau=_rect_smooth_tau(overrides),
                )
                grad = np.asarray(geom["barrier_grad"], dtype=np.float32).reshape(2)
                hess = np.asarray(geom["barrier_hess"], dtype=np.float32).reshape(2, 2)
                h_term = float(geom["barrier_h"])
                hdot_term = float(np.dot(grad, vel))
                const_term = float(vel @ hess @ vel)
                a_row = (-grad).astype(np.float32)
                b_row = const_term + (k1 * hocbf_gamma_hdot) * hdot_term + (k0 * hocbf_gamma_h) * h_term
                b_row -= _robust_margin(a_row, disturbance_accel_max, robust_cbf)
                A_cbf_rows.append(a_row)
                b_cbf_rows.append(float(b_row))

        if boundary_cbf and world_size > 0.0:
            xmin = -world_size + boundary_margin
            xmax = world_size - boundary_margin
            ymin = -world_size + boundary_margin
            ymax = world_size - boundary_margin
            for axis, upper in ((0, False), (0, True), (1, False), (1, True)):
                pos = float(state_i.position[axis])
                vel = float(state_i.velocity[axis])
                if upper:
                    h0 = float((xmax if axis == 0 else ymax) - pos)
                    h0_dot = float(-vel)
                    a_row = np.zeros((2,), dtype=np.float32)
                    a_row[axis] = 1.0
                else:
                    h0 = float(pos - (xmin if axis == 0 else ymin))
                    h0_dot = float(vel)
                    a_row = np.zeros((2,), dtype=np.float32)
                    a_row[axis] = -1.0

                if cbf_mode == "distributed_gcbfplus":
                    alpha0 = k0 * hocbf_gamma_h
                    alpha1 = k1 * hocbf_gamma_hdot
                    h1 = h0_dot + alpha0 * h0
                    lf_h1 = alpha0 * h0_dot
                    b_row = lf_h1 + alpha1 * h1
                else:
                    b_row = (k1 * hocbf_gamma_hdot) * h0_dot + (k0 * hocbf_gamma_h) * h0
                b_row -= _robust_margin(a_row, disturbance_accel_max, robust_cbf)
                A_cbf_rows.append(a_row)
                b_cbf_rows.append(float(b_row))

        if A_cbf_rows:
            A_cbf = np.stack(A_cbf_rows, axis=0).astype(np.float32)
            b_cbf = np.asarray(b_cbf_rows, dtype=np.float32)
        else:
            A_cbf = np.zeros((0, 2), dtype=np.float32)
            b_cbf = np.zeros((0,), dtype=np.float32)

        if use_clf:
            # Soft CLF (target-style): velocity tracking to desired goal-directed speed.
            clf_k = float(np.asarray(qp_param.clf_k).reshape(-1)[0])
            if "clf_v_des_vector" in overrides:
                v_des = np.asarray(overrides["clf_v_des_vector"], dtype=np.float32).reshape(2)
            else:
                goal_vec = (state_i.goal - state_i.position).astype(np.float32)
                goal_norm = float(np.linalg.norm(goal_vec))
                if goal_norm > 1e-6:
                    goal_dir = goal_vec / goal_norm
                else:
                    vel_norm = float(np.linalg.norm(state_i.velocity))
                    if vel_norm > 1e-6:
                        goal_dir = state_i.velocity / vel_norm
                    else:
                        goal_dir = np.array([1.0, 0.0], dtype=np.float32)

                v_des_speed = float(
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
                )
                slow_radius = float(overrides.get("slow_radius", 0.0))
                if slow_radius > 0.0:
                    speed_scale = float(np.clip(goal_norm / slow_radius, 0.0, 1.0))
                    min_speed = float(overrides.get("goal_stop_min_speed", 0.0))
                    v_des_speed = max(min_speed, v_des_speed * speed_scale)
                v_des = v_des_speed * goal_dir
            v_err = state_i.velocity - v_des
            V = float(0.5 * np.dot(v_err, v_err))
            A_clf = v_err.reshape(1, 2).astype(np.float32)
            b_clf = np.asarray([-clf_k * V], dtype=np.float32)
            w_clf = float(np.asarray(qp_param.w_clf).reshape(-1)[0])
        else:
            A_clf = np.zeros((0, 2), dtype=np.float32)
            b_clf = np.zeros((0,), dtype=np.float32)
            w_clf = 0.0

        H_mat = np.asarray(qp_param.H_mat if H_override is None else H_override, dtype=np.float32).reshape(2, 2)
        H_mat = 0.5 * (H_mat + H_mat.T)
        H_mat += 1e-6 * np.eye(2, dtype=np.float32)
        if f_lin_override is not None:
            f_lin = np.asarray(f_lin_override, dtype=np.float32).reshape(2)
        else:
            f_lin = np.asarray(qp_param.f_lin, dtype=np.float32).reshape(2)

        return QPProblem(
            H_mat=H_mat,
            f_lin=f_lin,
            w_clf=w_clf,
            w_cbf=float(np.asarray(qp_param.w_cbf).reshape(-1)[0] if "w_cbf" not in overrides else overrides["w_cbf"]),
            cbf_slack_max=float(
                np.asarray(qp_param.cbf_slack_max).reshape(-1)[0]
                if "cbf_slack_max" not in overrides
                else overrides["cbf_slack_max"]
            ),
            A_cbf=A_cbf,
            b_cbf=b_cbf,
            A_clf=A_clf,
            b_clf=b_clf,
            u_min=u_min.copy(),
            u_max=u_max.copy(),
            delta_min=0.0,
        )


def _build_hocbf54_row(
    p_rel: np.ndarray,
    v_rel: np.ndarray,
    safe_distance: float,
    u_max: float,
    share: float,
    eps: float,
) -> tuple[np.ndarray, float, float, float]:
    """
    GCBF+-style reciprocal CBF row from Eq.(54)-type barrier:
        h = sqrt(4*u_max*(||p_rel|| - safe_distance)) + n^T v_rel
    with local affine constraint in u_i:
        A u_i <= const + k1*hdot_term + k0*h_term
    """
    p_rel = np.asarray(p_rel, dtype=np.float32).reshape(2)
    v_rel = np.asarray(v_rel, dtype=np.float32).reshape(2)
    d = float(np.linalg.norm(p_rel))
    d_safe = float(max(safe_distance, eps))
    d_eff = float(max(d, d_safe + eps))
    n = (p_rel / max(d_eff, eps)).astype(np.float32)
    d_dot = float(np.dot(n, v_rel))

    gap = float(max(d_eff - d_safe, eps))
    sqrt_term = float(np.sqrt(max(4.0 * max(u_max, eps) * gap, eps)))
    h = float(sqrt_term + d_dot)

    kappa = float((2.0 * max(u_max, eps) / max(sqrt_term, eps)) * d_dot)
    n_dot_v = float((float(np.dot(v_rel, v_rel)) - d_dot * d_dot) / max(d_eff, eps))

    a_row = (-n).astype(np.float32)
    h_term = float(share * h)
    hdot_term = float(share * (kappa + n_dot_v))
    const_term = 0.0
    return a_row, h_term, hdot_term, const_term
