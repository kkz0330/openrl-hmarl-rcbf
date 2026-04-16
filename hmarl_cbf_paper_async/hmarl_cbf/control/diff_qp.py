from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
except ImportError:  # pragma: no cover - optional backend
    cp = None  # type: ignore[assignment]
    CvxpyLayer = None  # type: ignore[assignment]

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - optional backend
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc, assignment]

from hmarl_cbf.env.obstacles import obstacle_cbf_geometries, normalize_obstacle, rect_corner_margin_geometry
from hmarl_cbf.types import AgentState, QPParam

SPD_EPS = 1e-5


def _rect_corner_extra_margin(
    state_i: AgentState,
    obs_norm: Dict[str, Any],
    geom: Dict[str, Any],
    *,
    enabled: bool,
    margin_max: float,
    threshold: float,
    speed_min: float,
    alignment_power: float,
) -> float:
    if str(obs_norm.get("type", "")).strip().lower() != "rect":
        return 0.0
    if not enabled:
        return 0.0
    margin_max = float(max(0.0, margin_max))
    if margin_max <= 0.0:
        return 0.0
    threshold = float(max(1e-6, threshold))
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
    speed_min = float(max(0.0, speed_min))
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
    alignment_power = float(max(0.25, alignment_power))
    return float(margin_max * corner_proximity * (alignment**alignment_power))


def _rect_base_extra_margin(rect_base_margin_extra: float) -> float:
    return float(max(0.0, rect_base_margin_extra))


def _rect_dual_edge_proximity_distance(rect_dual_edge_proximity_distance: float, rect_corner_proximity_distance: float) -> float:
    return float(max(1e-6, rect_dual_edge_proximity_distance if rect_dual_edge_proximity_distance > 0.0 else rect_corner_proximity_distance))


@dataclass(slots=True)
class DiffConstraintConstants:
    """State-dependent constants; QP coefficients are produced by network outputs."""

    A_cbf: Tensor
    cbf_mode: str
    cbf_const: Tensor
    cbf_h: Tensor
    cbf_hdot: Tensor
    cbf_h0: Tensor
    cbf_h0dot: Tensor
    cbf_pv: Tensor
    cbf_v2: Tensor
    cbf_resp: Tensor
    A_clf: Tensor
    clf_V: Tensor
    u_min: Tensor
    u_max: Tensor


@dataclass(slots=True)
class DiffQPSolveResult:
    action: Tensor
    slack: Tensor
    cbf_slack: Tensor
    b_cbf: Tensor
    b_clf: Tensor


class TorchDifferentiableQPSolver:
    """
    Differentiable QP solver based on cvxpylayers.

    Supports gradients through:
    - full SPD H via parameterized Cholesky factor
    - f_lin
    - w_clf / w_cbf / cbf_slack_max
    - cbf_k0 / cbf_k1 (via b_cbf)
    - clf_k (via b_clf)
    """

    def __init__(
        self,
        action_dim: int = 2,
        ecos_max_iters: int = 500,
        scs_max_iters: int = 10_000,
        scs_eps: float = 1e-4,
    ) -> None:
        if torch is None or cp is None or CvxpyLayer is None:
            raise RuntimeError("TorchDifferentiableQPSolver requires torch, cvxpy and cvxpylayers")
        self.action_dim = action_dim
        self.ecos_max_iters = int(ecos_max_iters)
        self.scs_max_iters = int(scs_max_iters)
        self.scs_eps = float(scs_eps)
        self._cache: Dict[Tuple[int, int, int], CvxpyLayer] = {}

    def _get_layer(self, m_cbf: int, m_clf: int, n_u: int) -> CvxpyLayer:
        key = (m_cbf, m_clf, n_u)
        if key in self._cache:
            return self._cache[key]

        u = cp.Variable(n_u)
        delta = cp.Variable(1, nonneg=True)
        eps_cbf = cp.Variable(m_cbf, nonneg=True) if m_cbf > 0 else None

        H_sqrt = cp.Parameter((n_u, n_u))
        f_lin = cp.Parameter(n_u)
        w_clf = cp.Parameter(1, nonneg=True)
        w_cbf = cp.Parameter(1, nonneg=True)
        A_cbf = cp.Parameter((m_cbf, n_u))
        b_cbf = cp.Parameter(m_cbf)
        A_clf = cp.Parameter((m_clf, n_u))
        b_clf = cp.Parameter(m_clf)
        u_min = cp.Parameter(n_u)
        u_max = cp.Parameter(n_u)
        cbf_slack_max = cp.Parameter(1, nonneg=True)

        objective = (
            0.5 * cp.sum_squares(H_sqrt @ u)
            + f_lin @ u
            + cp.sum(cp.multiply(w_clf, delta))
        )
        if m_cbf > 0 and eps_cbf is not None:
            objective += cp.sum(cp.multiply(w_cbf, eps_cbf))

        constraints = [u >= u_min, u <= u_max, delta >= 0.0]
        if m_cbf > 0 and eps_cbf is not None:
            constraints.extend(
                [
                    A_cbf @ u <= b_cbf + eps_cbf,
                    eps_cbf <= cp.multiply(np.ones((m_cbf,), dtype=np.float32), cbf_slack_max),
                ]
            )
        if m_clf > 0:
            constraints.append(A_clf @ u <= b_clf + delta)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        if not problem.is_dpp():
            raise RuntimeError("Differentiable QP must be DPP-compliant")
        layer = CvxpyLayer(
            problem,
            parameters=[H_sqrt, f_lin, w_clf, w_cbf, A_cbf, b_cbf, A_clf, b_clf, u_min, u_max, cbf_slack_max],
            variables=[u, delta, eps_cbf] if m_cbf > 0 and eps_cbf is not None else [u, delta],
        )
        self._cache[key] = layer
        return layer

    @staticmethod
    def _project_spd_torch(H: Tensor) -> Tensor:
        H = 0.5 * (H + H.transpose(-1, -2))
        eigvals, eigvecs = torch.linalg.eigh(H)
        eigvals = torch.clamp(eigvals, min=SPD_EPS)
        return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)

    def solve(self, qp_param: QPParam, constants: DiffConstraintConstants) -> DiffQPSolveResult:
        if torch is None:
            raise RuntimeError("PyTorch is required")

        device = constants.A_cbf.device
        dtype = constants.A_cbf.dtype
        n_u = int(constants.A_cbf.shape[1])
        m_cbf = int(constants.A_cbf.shape[0])
        m_clf = int(constants.A_clf.shape[0])

        cbf_k0 = torch.reshape(torch.as_tensor(qp_param.cbf_k0, device=device, dtype=dtype), ())
        cbf_k1 = torch.reshape(torch.as_tensor(qp_param.cbf_k1, device=device, dtype=dtype), ())
        hocbf_gamma_h = (
            torch.reshape(torch.as_tensor(qp_param.hocbf_gamma_h, device=device, dtype=dtype), ())
            if qp_param.hocbf_gamma_h is not None
            else torch.tensor(1.0, device=device, dtype=dtype)
        )
        hocbf_gamma_hdot = (
            torch.reshape(torch.as_tensor(qp_param.hocbf_gamma_hdot, device=device, dtype=dtype), ())
            if qp_param.hocbf_gamma_hdot is not None
            else torch.tensor(1.0, device=device, dtype=dtype)
        )
        clf_k = torch.reshape(torch.as_tensor(qp_param.clf_k, device=device, dtype=dtype), ())
        if constants.cbf_mode == "distributed_gcbfplus":
            alpha0 = cbf_k0 * hocbf_gamma_h
            alpha1 = cbf_k1 * hocbf_gamma_hdot
            h1 = constants.cbf_h0dot + alpha0 * constants.cbf_h0
            lf_h1 = 2.0 * constants.cbf_v2 + 2.0 * alpha0 * constants.cbf_pv
            b_cbf = constants.cbf_resp * (lf_h1 + alpha1 * h1)
        else:
            b_cbf = (
                constants.cbf_const
                + (cbf_k1 * hocbf_gamma_hdot) * constants.cbf_hdot
                + (cbf_k0 * hocbf_gamma_h) * constants.cbf_h
            )
        b_clf = -clf_k * constants.clf_V

        layer = self._get_layer(m_cbf=m_cbf, m_clf=m_clf, n_u=n_u)
        H_mat = torch.as_tensor(qp_param.H_mat, device=device, dtype=dtype).reshape(n_u, n_u)
        H_mat = self._project_spd_torch(H_mat)
        H_sqrt = torch.linalg.cholesky(H_mat).transpose(-1, -2)
        f_lin = torch.as_tensor(qp_param.f_lin, device=device, dtype=dtype).reshape(n_u)
        w_clf = torch.as_tensor(qp_param.w_clf, device=device, dtype=dtype).reshape(1)
        w_cbf = torch.as_tensor(qp_param.w_cbf, device=device, dtype=dtype).reshape(1)
        cbf_slack_max = torch.as_tensor(qp_param.cbf_slack_max, device=device, dtype=dtype).reshape(1)

        try:
            outputs = layer(
                H_sqrt,
                f_lin,
                w_clf,
                w_cbf,
                constants.A_cbf,
                b_cbf.reshape(m_cbf),
                constants.A_clf,
                b_clf.reshape(m_clf),
                constants.u_min.reshape(n_u),
                constants.u_max.reshape(n_u),
                cbf_slack_max,
                solver_args={
                    "solve_method": "ECOS",
                    "max_iters": self.ecos_max_iters,
                },
            )
        except Exception:
            try:
                outputs = layer(
                    H_sqrt,
                    f_lin,
                    w_clf,
                    w_cbf,
                    constants.A_cbf,
                    b_cbf.reshape(m_cbf),
                    constants.A_clf,
                    b_clf.reshape(m_clf),
                    constants.u_min.reshape(n_u),
                    constants.u_max.reshape(n_u),
                    cbf_slack_max,
                    solver_args={
                        "solve_method": "SCS",
                        "max_iters": self.scs_max_iters,
                        "eps": self.scs_eps,
                    },
                )
            except Exception:
                action = -torch.linalg.pinv(H_mat) @ f_lin
                action = torch.maximum(torch.minimum(action, constants.u_max.reshape(n_u)), constants.u_min.reshape(n_u))
                slack = torch.zeros((1,), dtype=dtype, device=device)
                cbf_slack = torch.zeros((m_cbf,), dtype=dtype, device=device)
                return DiffQPSolveResult(
                    action=action.reshape(n_u),
                    slack=slack.reshape(1),
                    cbf_slack=cbf_slack.reshape(m_cbf),
                    b_cbf=b_cbf.reshape(m_cbf),
                    b_clf=b_clf.reshape(m_clf),
                )
        action = outputs[0]
        slack = outputs[1]
        cbf_slack = outputs[2] if len(outputs) > 2 else torch.zeros((m_cbf,), dtype=dtype, device=device)
        return DiffQPSolveResult(
            action=action.reshape(n_u),
            slack=slack.reshape(1),
            cbf_slack=cbf_slack.reshape(m_cbf),
            b_cbf=b_cbf.reshape(m_cbf),
            b_clf=b_clf.reshape(m_clf),
        )


def build_diff_constraint_constants(
    state_i: AgentState,
    neighbors: List[AgentState],
    obstacles: List[Dict[str, np.ndarray | float]],
    d_min_agent: float = 0.6,
    d_safe_obs: float = 0.6,
    cbf_mode: str = "distributed_ecbf",
    cbf_u_max: float = 1.0,
    cbf_share_agent: float = 0.5,
    cbf_share_obs: float = 1.0,
    cbf_eps: float = 1e-4,
    boundary_cbf: bool = False,
    world_size: float = 0.0,
    boundary_margin: float = 0.0,
    rect_corner_margin_enabled: bool = False,
    rect_base_margin_extra: float = 0.0,
    rect_corner_margin_max: float = 0.0,
    rect_corner_proximity_distance: float = 0.4,
    rect_corner_speed_min: float = 0.05,
    rect_corner_alignment_power: float = 1.0,
    rect_dual_edge_cbf_enabled: bool = False,
    rect_dual_edge_proximity_distance: float = 0.0,
    clf_v_des_speed: float = 0.8,
    u_min: np.ndarray | List[float] = (-1.0, -1.0),
    u_max: np.ndarray | List[float] = (1.0, 1.0),
    device: str | None = None,
) -> DiffConstraintConstants:
    if torch is None:
        raise RuntimeError("PyTorch is required for differentiable constraints")

    dev = torch.device(device or "cpu")
    dtype = torch.float32
    p_i = torch.tensor(state_i.position, dtype=dtype, device=dev)
    v_i = torch.tensor(state_i.velocity, dtype=dtype, device=dev)

    A_rows: List[Tensor] = []
    cbf_const_terms: List[Tensor] = []
    cbf_h_terms: List[Tensor] = []
    cbf_hdot_terms: List[Tensor] = []
    cbf_h0_terms: List[Tensor] = []
    cbf_h0dot_terms: List[Tensor] = []
    cbf_pv_terms: List[Tensor] = []
    cbf_v2_terms: List[Tensor] = []
    cbf_resp_terms: List[Tensor] = []

    for state_j in neighbors:
        p_j = torch.tensor(state_j.position, dtype=dtype, device=dev)
        v_j = torch.tensor(state_j.velocity, dtype=dtype, device=dev)
        p_rel = p_i - p_j
        v_rel = v_i - v_j
        if cbf_mode == "distributed_hocbf54":
            a_row, h_term, hdot_term, const_term = _build_hocbf54_row_torch(
                p_rel=p_rel,
                v_rel=v_rel,
                safe_distance=float(max(d_min_agent, cbf_eps)),
                u_max=float(cbf_u_max),
                share=float(cbf_share_agent),
                eps=float(cbf_eps),
                dtype=dtype,
                device=dev,
            )
            A_rows.append(a_row)
            cbf_const_terms.append(const_term)
            cbf_h_terms.append(h_term)
            cbf_hdot_terms.append(hdot_term)
            cbf_h0_terms.append(torch.zeros((), dtype=dtype, device=dev))
            cbf_h0dot_terms.append(torch.zeros((), dtype=dtype, device=dev))
            cbf_pv_terms.append(torch.zeros((), dtype=dtype, device=dev))
            cbf_v2_terms.append(torch.zeros((), dtype=dtype, device=dev))
            cbf_resp_terms.append(torch.ones((), dtype=dtype, device=dev))
        elif cbf_mode == "distributed_gcbfplus":
            h0 = torch.dot(p_rel, p_rel) - torch.tensor(float(d_min_agent**2), dtype=dtype, device=dev)
            pv = torch.dot(p_rel, v_rel)
            h0_dot = 2.0 * pv
            v2 = torch.dot(v_rel, v_rel)
            A_rows.append(-2.0 * p_rel)
            cbf_const_terms.append(torch.zeros((), dtype=dtype, device=dev))
            cbf_h_terms.append(torch.zeros((), dtype=dtype, device=dev))
            cbf_hdot_terms.append(torch.zeros((), dtype=dtype, device=dev))
            cbf_h0_terms.append(h0)
            cbf_h0dot_terms.append(h0_dot)
            cbf_pv_terms.append(pv)
            cbf_v2_terms.append(v2)
            cbf_resp_terms.append(torch.tensor(float(cbf_share_agent), dtype=dtype, device=dev))
        else:
            h = torch.dot(p_rel, p_rel) - torch.tensor(float(d_min_agent**2), dtype=dtype, device=dev)
            h_dot = 2.0 * torch.dot(p_rel, v_rel)
            cbf_const = 2.0 * torch.dot(v_rel, v_rel)
            A_rows.append(-2.0 * p_rel)
            cbf_const_terms.append(cbf_const)
            cbf_h_terms.append(h)
            cbf_hdot_terms.append(h_dot)
            cbf_h0_terms.append(torch.zeros((), dtype=dtype, device=dev))
            cbf_h0dot_terms.append(torch.zeros((), dtype=dtype, device=dev))
            cbf_pv_terms.append(torch.zeros((), dtype=dtype, device=dev))
            cbf_v2_terms.append(torch.zeros((), dtype=dtype, device=dev))
            cbf_resp_terms.append(torch.ones((), dtype=dtype, device=dev))

    for obs in obstacles:
        obs_norm = normalize_obstacle(obs)
        center = torch.tensor(np.asarray(obs_norm["center"], dtype=np.float32).reshape(2), dtype=dtype, device=dev)
        if obs_norm["type"] in ("circle", "point"):
            radius = float(obs_norm["radius"])
            p_rel = p_i - center
            if cbf_mode == "distributed_hocbf54":
                a_row, h_term, hdot_term, const_term = _build_hocbf54_row_torch(
                    p_rel=p_rel,
                    v_rel=v_i,
                    safe_distance=float(max(radius + d_safe_obs, cbf_eps)),
                    u_max=float(cbf_u_max),
                    share=float(cbf_share_obs),
                    eps=float(cbf_eps),
                    dtype=dtype,
                    device=dev,
                )
                A_rows.append(a_row)
                cbf_const_terms.append(const_term)
                cbf_h_terms.append(h_term)
                cbf_hdot_terms.append(hdot_term)
                cbf_h0_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_h0dot_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_pv_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_v2_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_resp_terms.append(torch.ones((), dtype=dtype, device=dev))
            elif cbf_mode == "distributed_gcbfplus":
                h0 = torch.dot(p_rel, p_rel) - torch.tensor(float((radius + d_safe_obs) ** 2), dtype=dtype, device=dev)
                pv = torch.dot(p_rel, v_i)
                h0_dot = 2.0 * pv
                v2 = torch.dot(v_i, v_i)
                A_rows.append(-2.0 * p_rel)
                cbf_const_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_h_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_hdot_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_h0_terms.append(h0)
                cbf_h0dot_terms.append(h0_dot)
                cbf_pv_terms.append(pv)
                cbf_v2_terms.append(v2)
                cbf_resp_terms.append(torch.tensor(float(cbf_share_obs), dtype=dtype, device=dev))
            else:
                h = torch.dot(p_rel, p_rel) - torch.tensor(float((radius + d_safe_obs) ** 2), dtype=dtype, device=dev)
                h_dot = 2.0 * torch.dot(p_rel, v_i)
                cbf_const = 2.0 * torch.dot(v_i, v_i)
                A_rows.append(-2.0 * p_rel)
                cbf_const_terms.append(cbf_const)
                cbf_h_terms.append(h)
                cbf_hdot_terms.append(h_dot)
                cbf_h0_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_h0dot_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_pv_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_v2_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_resp_terms.append(torch.ones((), dtype=dtype, device=dev))
        else:
            rect_geoms = obstacle_cbf_geometries(
                state_i.position,
                obs_norm,
                rect_dual_edge_enabled=bool(rect_dual_edge_cbf_enabled),
                rect_dual_edge_proximity_distance=_rect_dual_edge_proximity_distance(
                    float(rect_dual_edge_proximity_distance),
                    float(rect_corner_proximity_distance),
                ),
            )
            for geom in rect_geoms:
                offset_np = np.asarray(geom["offset"], dtype=np.float32).reshape(2)
                offset = torch.tensor(offset_np, dtype=dtype, device=dev)
                sign = float(geom["sign"])
                d_safe_obs_eff = d_safe_obs + _rect_base_extra_margin(rect_base_margin_extra)
                if str(geom.get("face_role", "primary")).strip().lower() != "secondary":
                    d_safe_obs_eff += _rect_corner_extra_margin(
                        state_i=state_i,
                        obs_norm=obs_norm,
                        geom=geom,
                        enabled=rect_corner_margin_enabled,
                        margin_max=rect_corner_margin_max,
                        threshold=rect_corner_proximity_distance,
                        speed_min=rect_corner_speed_min,
                        alignment_power=rect_corner_alignment_power,
                    )
                if cbf_mode == "distributed_gcbfplus":
                    h0 = sign * torch.dot(offset, offset) - torch.tensor(float(d_safe_obs_eff**2), dtype=dtype, device=dev)
                    pv = torch.dot(offset, v_i)
                    h0_dot = 2.0 * sign * pv
                    v2 = torch.dot(v_i, v_i)
                    A_rows.append(-2.0 * sign * offset)
                    cbf_const_terms.append(torch.zeros((), dtype=dtype, device=dev))
                    cbf_h_terms.append(torch.zeros((), dtype=dtype, device=dev))
                    cbf_hdot_terms.append(torch.zeros((), dtype=dtype, device=dev))
                    cbf_h0_terms.append(h0)
                    cbf_h0dot_terms.append(h0_dot)
                    cbf_pv_terms.append(pv)
                    cbf_v2_terms.append(sign * v2)
                    cbf_resp_terms.append(torch.tensor(float(cbf_share_obs), dtype=dtype, device=dev))
                else:
                    h = sign * torch.dot(offset, offset) - torch.tensor(float(d_safe_obs_eff**2), dtype=dtype, device=dev)
                    h_dot = 2.0 * sign * torch.dot(offset, v_i)
                    cbf_const = 2.0 * sign * torch.dot(v_i, v_i)
                    A_rows.append(-2.0 * sign * offset)
                    cbf_const_terms.append(cbf_const)
                    cbf_h_terms.append(h)
                    cbf_hdot_terms.append(h_dot)
                    cbf_h0_terms.append(torch.zeros((), dtype=dtype, device=dev))
                    cbf_h0dot_terms.append(torch.zeros((), dtype=dtype, device=dev))
                    cbf_pv_terms.append(torch.zeros((), dtype=dtype, device=dev))
                    cbf_v2_terms.append(torch.zeros((), dtype=dtype, device=dev))
                    cbf_resp_terms.append(torch.ones((), dtype=dtype, device=dev))

    if boundary_cbf and world_size > 0.0:
        xmin = float(-world_size + boundary_margin)
        xmax = float(world_size - boundary_margin)
        ymin = float(-world_size + boundary_margin)
        ymax = float(world_size - boundary_margin)
        bounds = (
            (0, False, xmin),
            (0, True, xmax),
            (1, False, ymin),
            (1, True, ymax),
        )
        for axis, upper, bound in bounds:
            pos = p_i[axis]
            vel = v_i[axis]
            a_row = torch.zeros((2,), dtype=dtype, device=dev)
            if upper:
                h0 = torch.tensor(float(bound), dtype=dtype, device=dev) - pos
                h0_dot = -vel
                a_row[axis] = 1.0
            else:
                h0 = pos - torch.tensor(float(bound), dtype=dtype, device=dev)
                h0_dot = vel
                a_row[axis] = -1.0

            if cbf_mode == "distributed_gcbfplus":
                A_rows.append(a_row)
                cbf_const_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_h_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_hdot_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_h0_terms.append(h0)
                cbf_h0dot_terms.append(h0_dot)
                cbf_pv_terms.append(0.5 * h0_dot)
                cbf_v2_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_resp_terms.append(torch.ones((), dtype=dtype, device=dev))
            else:
                A_rows.append(a_row)
                cbf_const_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_h_terms.append(h0)
                cbf_hdot_terms.append(h0_dot)
                cbf_h0_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_h0dot_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_pv_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_v2_terms.append(torch.zeros((), dtype=dtype, device=dev))
                cbf_resp_terms.append(torch.ones((), dtype=dtype, device=dev))

    if len(A_rows) == 0:
        A_cbf = torch.zeros((1, 2), dtype=dtype, device=dev)
        cbf_const = torch.zeros((1,), dtype=dtype, device=dev)
        cbf_h = torch.ones((1,), dtype=dtype, device=dev)
        cbf_hdot = torch.zeros((1,), dtype=dtype, device=dev)
        cbf_h0 = torch.zeros((1,), dtype=dtype, device=dev)
        cbf_h0dot = torch.zeros((1,), dtype=dtype, device=dev)
        cbf_pv = torch.zeros((1,), dtype=dtype, device=dev)
        cbf_v2 = torch.zeros((1,), dtype=dtype, device=dev)
        cbf_resp = torch.ones((1,), dtype=dtype, device=dev)
    else:
        A_cbf = torch.stack(A_rows, dim=0).to(dtype=dtype, device=dev)
        cbf_const = torch.stack(cbf_const_terms, dim=0).to(dtype=dtype, device=dev)
        cbf_h = torch.stack(cbf_h_terms, dim=0).to(dtype=dtype, device=dev)
        cbf_hdot = torch.stack(cbf_hdot_terms, dim=0).to(dtype=dtype, device=dev)
        cbf_h0 = torch.stack(cbf_h0_terms, dim=0).to(dtype=dtype, device=dev)
        cbf_h0dot = torch.stack(cbf_h0dot_terms, dim=0).to(dtype=dtype, device=dev)
        cbf_pv = torch.stack(cbf_pv_terms, dim=0).to(dtype=dtype, device=dev)
        cbf_v2 = torch.stack(cbf_v2_terms, dim=0).to(dtype=dtype, device=dev)
        cbf_resp = torch.stack(cbf_resp_terms, dim=0).to(dtype=dtype, device=dev)

    goal = torch.tensor(state_i.goal, dtype=dtype, device=dev)
    goal_vec = goal - p_i
    goal_norm = torch.linalg.norm(goal_vec)
    if float(goal_norm.detach().cpu().item()) > 1e-6:
        goal_dir = goal_vec / goal_norm
    else:
        v_norm = torch.linalg.norm(v_i)
        if float(v_norm.detach().cpu().item()) > 1e-6:
            goal_dir = v_i / v_norm
        else:
            goal_dir = torch.tensor([1.0, 0.0], dtype=dtype, device=dev)

    v_des_speed = torch.tensor(float(clf_v_des_speed), dtype=dtype, device=dev)
    v_des = v_des_speed * goal_dir
    v_err = v_i - v_des
    V = 0.5 * torch.dot(v_err, v_err)
    A_clf = v_err.reshape(1, 2).to(dtype=dtype, device=dev)
    clf_V = V.reshape(1).to(dtype=dtype, device=dev)

    return DiffConstraintConstants(
        A_cbf=A_cbf,
        cbf_mode=str(cbf_mode),
        cbf_const=cbf_const,
        cbf_h=cbf_h,
        cbf_hdot=cbf_hdot,
        cbf_h0=cbf_h0,
        cbf_h0dot=cbf_h0dot,
        cbf_pv=cbf_pv,
        cbf_v2=cbf_v2,
        cbf_resp=cbf_resp,
        A_clf=A_clf,
        clf_V=clf_V,
        u_min=torch.tensor(np.asarray(u_min, dtype=np.float32).reshape(2), dtype=dtype, device=dev),
        u_max=torch.tensor(np.asarray(u_max, dtype=np.float32).reshape(2), dtype=dtype, device=dev),
    )


def _build_hocbf54_row_torch(
    p_rel: Tensor,
    v_rel: Tensor,
    safe_distance: float,
    u_max: float,
    share: float,
    eps: float,
    dtype: Any,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    eps_t = torch.tensor(float(eps), dtype=dtype, device=device)
    d_safe_t = torch.tensor(float(max(safe_distance, eps)), dtype=dtype, device=device)
    u_max_t = torch.tensor(float(max(u_max, eps)), dtype=dtype, device=device)
    share_t = torch.tensor(float(share), dtype=dtype, device=device)

    d = torch.linalg.norm(p_rel)
    d_eff = torch.maximum(d, d_safe_t + eps_t)
    n = p_rel / torch.maximum(d_eff, eps_t)
    d_dot = torch.dot(n, v_rel)

    gap = torch.maximum(d_eff - d_safe_t, eps_t)
    sqrt_term = torch.sqrt(torch.maximum(4.0 * u_max_t * gap, eps_t))
    h = sqrt_term + d_dot

    kappa = (2.0 * u_max_t / torch.maximum(sqrt_term, eps_t)) * d_dot
    v_rel_sq = torch.dot(v_rel, v_rel)
    n_dot_v = (v_rel_sq - d_dot * d_dot) / torch.maximum(d_eff, eps_t)

    a_row = -n
    h_term = share_t * h
    hdot_term = share_t * (kappa + n_dot_v)
    const_term = torch.zeros((), dtype=dtype, device=device)
    return a_row, h_term, hdot_term, const_term
