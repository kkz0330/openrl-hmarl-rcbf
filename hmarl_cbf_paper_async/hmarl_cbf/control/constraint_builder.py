from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from hmarl_cbf.env.obstacles import obstacle_contact_geometry, normalize_obstacle, rect_corner_margin_geometry
from hmarl_cbf.types import AgentState, QPParam, QPProblem


def _rect_corner_extra_margin(
    state_i: AgentState,
    obs_norm: Dict[str, Any],
    geom: Dict[str, Any],
    overrides: Dict[str, Any],
) -> float:
    if str(obs_norm.get("type", "")).strip().lower() != "rect":
        return 0.0
    if not bool(overrides.get("rect_corner_margin_enabled", False)):
        return 0.0
    margin_max = float(max(0.0, overrides.get("rect_corner_margin_max", 0.0)))
    if margin_max <= 0.0:
        return 0.0
    threshold = float(max(1e-6, overrides.get("rect_corner_proximity_distance", 0.4)))
    corner_geom = rect_corner_margin_geometry(
        obs_norm,
        geom.get("closest_point_local", geom["closest_point"]),
        threshold=threshold,
    )
    corner_proximity = float(corner_geom["corner_proximity"])
    if corner_proximity <= 0.0:
        return 0.0

    vel = np.asarray(state_i.velocity, dtype=np.float32).reshape(2)
    speed = float(np.linalg.norm(vel))
    speed_min = float(max(0.0, overrides.get("rect_corner_speed_min", 0.05)))
    if speed <= speed_min:
        return 0.0

    pos = np.asarray(state_i.position, dtype=np.float32).reshape(2)
    nearest_corner = np.asarray(corner_geom["nearest_corner"], dtype=np.float32).reshape(2)
    to_corner = nearest_corner - pos
    dist = float(np.linalg.norm(to_corner))
    if dist <= 1e-8:
        alignment = 1.0
    else:
        alignment = max(0.0, float(np.dot(vel / speed, to_corner / dist)))
    if alignment <= 0.0:
        return 0.0
    alignment_power = float(max(0.25, overrides.get("rect_corner_alignment_power", 1.0)))
    return float(margin_max * corner_proximity * (alignment**alignment_power))


def _rect_base_extra_margin(overrides: Dict[str, Any]) -> float:
    return float(max(0.0, overrides.get("rect_base_margin_extra", 0.0)))


class ConstraintBuilder:
    """Builds distributed hard-CBF and soft-CLF constraints for one agent."""

    def __init__(
        self,
        d_min_agent: float = 0.6,
        d_safe_obs: float = 0.6,
        u_min: np.ndarray | list[float] = (-1.0, -1.0),
        u_max: np.ndarray | list[float] = (1.0, 1.0),
    ) -> None:
        self.d_min_agent = float(d_min_agent)
        self.d_safe_obs = float(d_safe_obs)
        self.u_min = np.asarray(u_min, dtype=np.float32).reshape(2)
        self.u_max = np.asarray(u_max, dtype=np.float32).reshape(2)

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
        d_min_agent = float(overrides.get("d_min_agent", self.d_min_agent))
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
        boundary_cbf = bool(overrides.get("boundary_cbf", False))
        world_size = float(overrides.get("world_size", 0.0))
        boundary_margin = float(overrides.get("boundary_margin", 0.0))

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
            elif cbf_mode == "distributed_gcbfplus":
                # GCBF+ dec-share style for double-integrator pairwise barrier:
                # h0 = ||p_rel||^2 - d_safe^2
                # h1 = h0_dot + alpha0 * h0, with h0_dot = 2 p_rel^T v_rel
                # -L_g h1 u_i <= resp * (L_f h1 + alpha1 * h1)
                h0 = float(np.dot(p_rel, p_rel) - d_min_agent**2)
                pv = float(np.dot(p_rel, v_rel))
                h0_dot = 2.0 * pv
                alpha0 = k0 * hocbf_gamma_h
                alpha1 = k1 * hocbf_gamma_hdot
                h1 = h0_dot + alpha0 * h0
                lf_h1 = 2.0 * float(np.dot(v_rel, v_rel)) + 2.0 * alpha0 * pv
                a_row = (-2.0 * p_rel).astype(np.float32)  # -L_g h1 wrt u_i
                b_row = cbf_share_agent * (lf_h1 + alpha1 * h1)
                A_cbf_rows.append(a_row)
                b_cbf_rows.append(float(b_row))
                continue
            else:
                # ECBF-like linearization wrt u_i, distributed assumption on u_j.
                h_term = float(np.dot(p_rel, p_rel) - d_min_agent**2)
                hdot_term = float(2.0 * np.dot(p_rel, v_rel))
                const_term = 2.0 * float(np.dot(v_rel, v_rel))
                a_row = (-2.0 * p_rel).astype(np.float32)
            b_row = const_term + (k1 * hocbf_gamma_hdot) * hdot_term + (k0 * hocbf_gamma_h) * h_term
            A_cbf_rows.append(a_row)
            b_cbf_rows.append(float(b_row))

        # Agent-obstacle CBF (circle / point / rect obstacles).
        for obs in obstacles:
            obs_norm = normalize_obstacle(obs)
            center = np.asarray(obs_norm["center"], dtype=np.float32).reshape(2)
            if obs_norm["type"] in ("circle", "point"):
                radius = float(obs_norm["radius"])
                p_rel = state_i.position - center
                if cbf_mode == "distributed_hocbf54":
                    a_row, h_term, hdot_term, const_term = _build_hocbf54_row(
                        p_rel=p_rel,
                        v_rel=state_i.velocity,
                        safe_distance=max(radius + d_safe_obs, 1e-4),
                        u_max=cbf_u_max,
                        share=cbf_share_obs,
                        eps=cbf_eps,
                    )
                elif cbf_mode == "distributed_gcbfplus":
                    h0 = float(np.dot(p_rel, p_rel) - (radius + d_safe_obs) ** 2)
                    pv = float(np.dot(p_rel, state_i.velocity))
                    h0_dot = 2.0 * pv
                    alpha0 = k0 * hocbf_gamma_h
                    alpha1 = k1 * hocbf_gamma_hdot
                    h1 = h0_dot + alpha0 * h0
                    lf_h1 = 2.0 * float(np.dot(state_i.velocity, state_i.velocity)) + 2.0 * alpha0 * pv
                    a_row = (-2.0 * p_rel).astype(np.float32)
                    b_row = cbf_share_obs * (lf_h1 + alpha1 * h1)
                    A_cbf_rows.append(a_row)
                    b_cbf_rows.append(float(b_row))
                    continue
                else:
                    h_term = float(np.dot(p_rel, p_rel) - (radius + d_safe_obs) ** 2)
                    hdot_term = float(2.0 * np.dot(p_rel, state_i.velocity))
                    const_term = 2.0 * float(np.dot(state_i.velocity, state_i.velocity))
                    a_row = (-2.0 * p_rel).astype(np.float32)
            else:
                geom = obstacle_contact_geometry(state_i.position, obs_norm)
                offset = np.asarray(geom["offset"], dtype=np.float32).reshape(2)
                sign = float(geom["sign"])
                vel = np.asarray(state_i.velocity, dtype=np.float32).reshape(2)
                d_safe_obs_eff = d_safe_obs + _rect_base_extra_margin(overrides) + _rect_corner_extra_margin(
                    state_i=state_i,
                    obs_norm=obs_norm,
                    geom=geom,
                    overrides=overrides,
                )
                if cbf_mode == "distributed_gcbfplus":
                    h0 = float(sign * np.dot(offset, offset) - d_safe_obs_eff**2)
                    pv = float(np.dot(offset, vel))
                    h0_dot = 2.0 * sign * pv
                    alpha0 = k0 * hocbf_gamma_h
                    alpha1 = k1 * hocbf_gamma_hdot
                    h1 = h0_dot + alpha0 * h0
                    lf_h1 = 2.0 * sign * float(np.dot(vel, vel)) + 2.0 * alpha0 * sign * pv
                    a_row = (-2.0 * sign * offset).astype(np.float32)
                    b_row = cbf_share_obs * (lf_h1 + alpha1 * h1)
                    A_cbf_rows.append(a_row)
                    b_cbf_rows.append(float(b_row))
                    continue
                h_term = float(sign * np.dot(offset, offset) - d_safe_obs_eff**2)
                hdot_term = float(2.0 * sign * np.dot(offset, vel))
                const_term = 2.0 * sign * float(np.dot(vel, vel))
                a_row = (-2.0 * sign * offset).astype(np.float32)
            b_row = const_term + (k1 * hocbf_gamma_hdot) * hdot_term + (k0 * hocbf_gamma_h) * h_term
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
                A_cbf_rows.append(a_row)
                b_cbf_rows.append(float(b_row))

        if A_cbf_rows:
            A_cbf = np.stack(A_cbf_rows, axis=0).astype(np.float32)
            b_cbf = np.asarray(b_cbf_rows, dtype=np.float32)
        else:
            A_cbf = np.zeros((0, 2), dtype=np.float32)
            b_cbf = np.zeros((0,), dtype=np.float32)

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
            w_clf=float(np.asarray(qp_param.w_clf).reshape(-1)[0]),
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
