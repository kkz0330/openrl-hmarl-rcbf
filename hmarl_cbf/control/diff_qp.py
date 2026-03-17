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

from hmarl_cbf.types import AgentState, QPParam


@dataclass(slots=True)
class DiffConstraintConstants:
    """State-dependent constants; QP coefficients are produced by network outputs."""

    A_cbf: Tensor
    cbf_const: Tensor
    cbf_h: Tensor
    cbf_hdot: Tensor
    A_clf: Tensor
    clf_V: Tensor
    u_min: Tensor
    u_max: Tensor


@dataclass(slots=True)
class DiffQPSolveResult:
    action: Tensor
    slack: Tensor
    b_cbf: Tensor
    b_clf: Tensor


class TorchDifferentiableQPSolver:
    """
    Differentiable QP solver based on cvxpylayers.

    Supports gradients through:
    - f_lin (or u_ref fallback)
    - r_diag
    - w_clf
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

        r_diag = cp.Parameter(n_u, nonneg=True)
        f_lin = cp.Parameter(n_u)
        w_clf = cp.Parameter(1, nonneg=True)
        A_cbf = cp.Parameter((m_cbf, n_u))
        b_cbf = cp.Parameter(m_cbf)
        A_clf = cp.Parameter((m_clf, n_u))
        b_clf = cp.Parameter(m_clf)
        u_min = cp.Parameter(n_u)
        u_max = cp.Parameter(n_u)

        # DPP-compliant diagonal-quadratic form:
        # 0.5 * u^T diag(r_diag) u + f_lin^T u + w_clf * delta
        objective = (
            0.5 * cp.sum(cp.multiply(r_diag, cp.square(u)))
            + f_lin @ u
            + cp.sum(cp.multiply(w_clf, delta))
        )
        constraints = [
            A_cbf @ u <= b_cbf,
            A_clf @ u <= b_clf + delta,
            u >= u_min,
            u <= u_max,
            delta >= 0.0,
        ]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        if not problem.is_dpp():
            raise RuntimeError("Differentiable QP must be DPP-compliant")
        layer = CvxpyLayer(
            problem,
            parameters=[r_diag, f_lin, w_clf, A_cbf, b_cbf, A_clf, b_clf, u_min, u_max],
            variables=[u, delta],
        )
        self._cache[key] = layer
        return layer

    def solve(self, qp_param: QPParam, constants: DiffConstraintConstants) -> DiffQPSolveResult:
        if torch is None:
            raise RuntimeError("PyTorch is required")

        device = constants.A_cbf.device
        dtype = constants.A_cbf.dtype
        n_u = int(constants.A_cbf.shape[1])
        m_cbf = int(constants.A_cbf.shape[0])
        m_clf = int(constants.A_clf.shape[0])

        # Keep these operations in torch graph for KKT backward.
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
        b_cbf = (
            constants.cbf_const
            + (cbf_k1 * hocbf_gamma_hdot) * constants.cbf_hdot
            + (cbf_k0 * hocbf_gamma_h) * constants.cbf_h
        )
        b_clf = -clf_k * constants.clf_V

        layer = self._get_layer(m_cbf=m_cbf, m_clf=m_clf, n_u=n_u)
        r_diag = torch.as_tensor(qp_param.r_diag, device=device, dtype=dtype).reshape(n_u)
        if qp_param.f_lin is not None:
            f_lin = torch.as_tensor(qp_param.f_lin, device=device, dtype=dtype).reshape(n_u)
        else:
            u_ref = torch.as_tensor(qp_param.u_ref, device=device, dtype=dtype).reshape(n_u)
            f_lin = -(r_diag * u_ref)
        w_clf = torch.as_tensor(qp_param.w_clf, device=device, dtype=dtype).reshape(1)

        try:
            action, slack = layer(
                r_diag,
                f_lin,
                w_clf,
                constants.A_cbf,
                b_cbf.reshape(m_cbf),
                constants.A_clf,
                b_clf.reshape(m_clf),
                constants.u_min.reshape(n_u),
                constants.u_max.reshape(n_u),
                solver_args={
                    "solve_method": "ECOS",
                    "max_iters": self.ecos_max_iters,
                },
            )
        except Exception:
            try:
                action, slack = layer(
                    r_diag,
                    f_lin,
                    w_clf,
                    constants.A_cbf,
                    b_cbf.reshape(m_cbf),
                    constants.A_clf,
                    b_clf.reshape(m_clf),
                    constants.u_min.reshape(n_u),
                    constants.u_max.reshape(n_u),
                    solver_args={
                        "solve_method": "SCS",
                        "max_iters": self.scs_max_iters,
                        "eps": self.scs_eps,
                    },
                )
            except Exception:
                action = -f_lin / torch.clamp(r_diag, min=1e-5)
                action = torch.maximum(torch.minimum(action, constants.u_max.reshape(n_u)), constants.u_min.reshape(n_u))
                slack = torch.zeros((1,), dtype=dtype, device=device)
        return DiffQPSolveResult(
            action=action.reshape(n_u),
            slack=slack.reshape(1),
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
    clf_v_des_speed: float = 0.8,
    u_min: np.ndarray | List[float] = (-1.0, -1.0),
    u_max: np.ndarray | List[float] = (1.0, 1.0),
    device: str | None = None,
) -> DiffConstraintConstants:
    """
    Builds state-dependent constants for a single-agent QP.

    This keeps CBF/CLF gain terms differentiable w.r.t network outputs.
    """
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
        else:
            h = torch.dot(p_rel, p_rel) - torch.tensor(float(d_min_agent**2), dtype=dtype, device=dev)
            h_dot = 2.0 * torch.dot(p_rel, v_rel)
            cbf_const = 2.0 * torch.dot(v_rel, v_rel)
            A_rows.append(-2.0 * p_rel)
            cbf_const_terms.append(cbf_const)
            cbf_h_terms.append(h)
            cbf_hdot_terms.append(h_dot)

    for obs in obstacles:
        center = torch.tensor(np.asarray(obs["center"], dtype=np.float32).reshape(2), dtype=dtype, device=dev)
        radius = float(obs["radius"])
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
        else:
            h = torch.dot(p_rel, p_rel) - torch.tensor(float((radius + d_safe_obs) ** 2), dtype=dtype, device=dev)
            h_dot = 2.0 * torch.dot(p_rel, v_i)
            cbf_const = 2.0 * torch.dot(v_i, v_i)
            A_rows.append(-2.0 * p_rel)
            cbf_const_terms.append(cbf_const)
            cbf_h_terms.append(h)
            cbf_hdot_terms.append(h_dot)

    if len(A_rows) == 0:
        # Ensure at least one CBF row so the layer signature is stable.
        A_cbf = torch.zeros((1, 2), dtype=dtype, device=dev)
        cbf_const = torch.zeros((1,), dtype=dtype, device=dev)
        cbf_h = torch.ones((1,), dtype=dtype, device=dev)
        cbf_hdot = torch.zeros((1,), dtype=dtype, device=dev)
    else:
        A_cbf = torch.stack(A_rows, dim=0).to(dtype=dtype, device=dev)
        cbf_const = torch.stack(cbf_const_terms, dim=0).to(dtype=dtype, device=dev)
        cbf_h = torch.stack(cbf_h_terms, dim=0).to(dtype=dtype, device=dev)
        cbf_hdot = torch.stack(cbf_hdot_terms, dim=0).to(dtype=dtype, device=dev)

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
        cbf_const=cbf_const,
        cbf_h=cbf_h,
        cbf_hdot=cbf_hdot,
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
