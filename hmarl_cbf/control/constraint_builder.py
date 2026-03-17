from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from hmarl_cbf.types import AgentState, QPParam, QPProblem


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
        u_ref_override: np.ndarray | None = None,
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
            else:
                # ECBF-like linearization wrt u_i, distributed assumption on u_j.
                h_term = float(np.dot(p_rel, p_rel) - d_min_agent**2)
                hdot_term = float(2.0 * np.dot(p_rel, v_rel))
                const_term = 2.0 * float(np.dot(v_rel, v_rel))
                a_row = (-2.0 * p_rel).astype(np.float32)
            b_row = const_term + (k1 * hocbf_gamma_hdot) * hdot_term + (k0 * hocbf_gamma_h) * h_term
            A_cbf_rows.append(a_row)
            b_cbf_rows.append(float(b_row))

        # Agent-obstacle CBF (circular obstacles).
        for obs in obstacles:
            center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
            radius = float(obs["radius"])
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
            else:
                h_term = float(np.dot(p_rel, p_rel) - (radius + d_safe_obs) ** 2)
                hdot_term = float(2.0 * np.dot(p_rel, state_i.velocity))
                const_term = 2.0 * float(np.dot(state_i.velocity, state_i.velocity))
                a_row = (-2.0 * p_rel).astype(np.float32)
            b_row = const_term + (k1 * hocbf_gamma_hdot) * hdot_term + (k0 * hocbf_gamma_h) * h_term
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
            v_des = v_des_speed * goal_dir
        v_err = state_i.velocity - v_des
        V = float(0.5 * np.dot(v_err, v_err))
        A_clf = v_err.reshape(1, 2).astype(np.float32)
        b_clf = np.asarray([-clf_k * V], dtype=np.float32)

        u_ref_eff = np.asarray(qp_param.u_ref if u_ref_override is None else u_ref_override, dtype=np.float32).reshape(2)
        r_diag = np.asarray(qp_param.r_diag, dtype=np.float32).reshape(2)
        if f_lin_override is not None:
            f_lin = np.asarray(f_lin_override, dtype=np.float32).reshape(2)
        elif qp_param.f_lin is not None:
            f_lin = np.asarray(qp_param.f_lin, dtype=np.float32).reshape(2)
        else:
            f_lin = -(r_diag * u_ref_eff)

        return QPProblem(
            u_ref=u_ref_eff,
            r_diag=r_diag,
            w_clf=float(np.asarray(qp_param.w_clf).reshape(-1)[0]),
            A_cbf=A_cbf,
            b_cbf=b_cbf,
            A_clf=A_clf,
            b_clf=b_clf,
            u_min=u_min.copy(),
            u_max=u_max.copy(),
            f_lin=f_lin,
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
