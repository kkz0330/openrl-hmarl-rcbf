from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
except ImportError:  # pragma: no cover - optional backend
    cp = None  # type: ignore[assignment]
    CvxpyLayer = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - import-safe fallback
    torch = None  # type: ignore[assignment]

from hmarl_cbf.types import QPProblem, QPSolution

SPD_EPS = 1e-5


def _symmetrize_spd_numpy(H: np.ndarray) -> np.ndarray:
    H = 0.5 * (H + H.T)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.maximum(eigvals, SPD_EPS)
    return (eigvecs @ np.diag(eigvals) @ eigvecs.T).astype(np.float32)


def _nominal_action(problem: QPProblem, action_dim: int) -> np.ndarray:
    H_mat = _symmetrize_spd_numpy(np.asarray(problem.H_mat, dtype=np.float32).reshape(action_dim, action_dim))
    f_lin = np.asarray(problem.f_lin, dtype=np.float32).reshape(-1)
    try:
        action = -np.linalg.solve(H_mat, f_lin).astype(np.float32)
    except np.linalg.LinAlgError:
        action = -np.linalg.pinv(H_mat).dot(f_lin).astype(np.float32)
    return np.clip(action, np.asarray(problem.u_min, dtype=np.float32), np.asarray(problem.u_max, dtype=np.float32))


def _top_constraint_indices(A: np.ndarray, b: np.ndarray, action: np.ndarray, top_k: int) -> np.ndarray:
    if A.size == 0 or top_k <= 0:
        return np.zeros((0,), dtype=np.int32)
    scores = (A @ action.reshape(-1) - b.reshape(-1)).astype(np.float32)
    order = np.argsort(scores)[::-1]
    keep = order[: min(int(top_k), int(order.shape[0]))]
    return np.sort(keep.astype(np.int32))


def solve_reduced_fallback_qp(
    problem: QPProblem,
    *,
    action_dim: int,
    ecos_max_iters: int,
    scs_max_iters: int,
    scs_eps: float,
    top_cbf: int = 4,
    top_clf: int = 1,
) -> QPSolution:
    action_guess = _nominal_action(problem, action_dim=action_dim)
    A_cbf = np.asarray(problem.A_cbf, dtype=np.float32).reshape(-1, action_dim)
    b_cbf = np.asarray(problem.b_cbf, dtype=np.float32).reshape(-1)
    A_clf = np.asarray(problem.A_clf, dtype=np.float32).reshape(-1, action_dim)
    b_clf = np.asarray(problem.b_clf, dtype=np.float32).reshape(-1)
    cbf_idx = _top_constraint_indices(A_cbf, b_cbf, action_guess, top_k=top_cbf)
    clf_idx = _top_constraint_indices(A_clf, b_clf, action_guess, top_k=top_clf)

    if cp is None or (A_cbf.shape[0] == 0 and A_clf.shape[0] == 0):
        clf_violation = 0.0
        if A_clf.size > 0:
            clf_violation = float(max(0.0, np.max(A_clf @ action_guess - b_clf)))
        cbf_slack = np.zeros((A_cbf.shape[0],), dtype=np.float32)
        if A_cbf.size > 0:
            cbf_slack = np.maximum(A_cbf @ action_guess - b_cbf, 0.0).astype(np.float32)
        return QPSolution(
            action=action_guess.astype(np.float32),
            slack=np.asarray([max(problem.delta_min, clf_violation)], dtype=np.float32),
            objective=np.asarray([0.0], dtype=np.float32),
            feasible=True,
            solver_status="reduced_stub",
            cbf_slack=cbf_slack,
        )

    H_mat = _symmetrize_spd_numpy(np.asarray(problem.H_mat, dtype=np.float32).reshape(action_dim, action_dim))
    f_lin = np.asarray(problem.f_lin, dtype=np.float32).reshape(action_dim)
    w_clf = float(np.asarray(problem.w_clf, dtype=np.float32).reshape(-1)[0])
    w_cbf = float(np.asarray(problem.w_cbf, dtype=np.float32).reshape(-1)[0])
    cbf_slack_max = float(np.asarray(problem.cbf_slack_max, dtype=np.float32).reshape(-1)[0])
    u_min = np.asarray(problem.u_min, dtype=np.float32).reshape(action_dim)
    u_max = np.asarray(problem.u_max, dtype=np.float32).reshape(action_dim)

    u = cp.Variable(action_dim)
    delta = cp.Variable(1, nonneg=True)
    objective = 0.5 * cp.quad_form(u, H_mat) + f_lin @ u + w_clf * cp.sum(delta)
    constraints = [u >= u_min, u <= u_max, delta >= float(problem.delta_min)]

    if cbf_idx.size > 0:
        eps = cp.Variable(cbf_idx.size, nonneg=True)
        objective += w_cbf * cp.sum(eps)
        constraints.append(A_cbf[cbf_idx] @ u <= b_cbf[cbf_idx] + eps)
        constraints.append(eps <= cbf_slack_max)
    else:
        eps = None

    if clf_idx.size > 0:
        constraints.append(A_clf[clf_idx] @ u <= b_clf[clf_idx] + delta)

    reduced_problem = cp.Problem(cp.Minimize(objective), constraints)
    status = "reduced_failed"
    solved = False
    for solver_name, solver_kwargs in (
        ("SCS", {"max_iters": int(scs_max_iters), "eps": float(scs_eps)}),
        ("ECOS", {"max_iters": int(ecos_max_iters)}),
    ):
        try:
            reduced_problem.solve(solver=solver_name, warm_start=True, **solver_kwargs)
        except Exception:
            continue
        if reduced_problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            status = f"reduced_{solver_name.lower()}"
            solved = True
            break

    if not solved or u.value is None:
        return QPSolution(
            action=action_guess.astype(np.float32),
            slack=np.asarray([0.0], dtype=np.float32),
            objective=np.asarray([0.0], dtype=np.float32),
            feasible=False,
            solver_status=status,
            cbf_slack=np.maximum(A_cbf @ action_guess - b_cbf, 0.0).astype(np.float32) if A_cbf.size > 0 else np.zeros((0,), dtype=np.float32),
        )

    action = np.asarray(u.value, dtype=np.float32).reshape(action_dim)
    action = np.clip(action, u_min, u_max)
    slack = np.asarray(delta.value if delta.value is not None else [0.0], dtype=np.float32).reshape(1)
    cbf_slack = np.zeros((A_cbf.shape[0],), dtype=np.float32)
    if eps is not None and eps.value is not None:
        cbf_slack[cbf_idx] = np.asarray(eps.value, dtype=np.float32).reshape(-1)
    return QPSolution(
        action=action,
        slack=slack,
        objective=np.asarray([float(reduced_problem.value) if reduced_problem.value is not None else 0.0], dtype=np.float32),
        feasible=True,
        solver_status=status,
        cbf_slack=cbf_slack,
    )


class DifferentiableQPSolver:
    """cvxpylayers-backed QP solver with a deterministic fallback path."""

    def __init__(
        self,
        action_dim: int = 2,
        use_stub_if_unavailable: bool = True,
        ecos_max_iters: int = 500,
        scs_max_iters: int = 10_000,
        scs_eps: float = 1e-4,
    ) -> None:
        self.action_dim = action_dim
        self.use_stub_if_unavailable = use_stub_if_unavailable
        self.ecos_max_iters = int(ecos_max_iters)
        self.scs_max_iters = int(scs_max_iters)
        self.scs_eps = float(scs_eps)
        self._cache: Dict[Tuple[int, int, int], CvxpyLayer] = {}
        self.backend = "cvxpylayers" if (cp is not None and CvxpyLayer is not None and torch is not None) else "stub"
        if self.backend == "stub" and not use_stub_if_unavailable:
            raise RuntimeError("cvxpylayers backend unavailable and stub fallback disabled")

    def solve(self, problem: QPProblem) -> QPSolution:
        if self.backend == "cvxpylayers":
            return self._solve_with_cvxpylayers(problem)
        return self._solve_stub(problem)

    def _solve_stub(self, problem: QPProblem) -> QPSolution:
        H_mat = _symmetrize_spd_numpy(np.asarray(problem.H_mat, dtype=np.float32).reshape(self.action_dim, self.action_dim))
        f_lin = np.asarray(problem.f_lin, dtype=np.float32).reshape(-1)
        try:
            action = -np.linalg.solve(H_mat, f_lin).astype(np.float32)
        except np.linalg.LinAlgError:
            action = -np.linalg.pinv(H_mat).dot(f_lin).astype(np.float32)
        action = np.clip(action, np.asarray(problem.u_min, dtype=np.float32), np.asarray(problem.u_max, dtype=np.float32))

        clf_violation = 0.0
        if np.asarray(problem.A_clf).size > 0:
            lhs = np.asarray(problem.A_clf, dtype=np.float32) @ action
            rhs = np.asarray(problem.b_clf, dtype=np.float32).reshape(-1)
            clf_violation = float(max(0.0, np.max(lhs - rhs)))
        slack = np.asarray([max(problem.delta_min, clf_violation)], dtype=np.float32)

        cbf_slack = np.zeros((int(np.asarray(problem.A_cbf).shape[0]),), dtype=np.float32)
        if np.asarray(problem.A_cbf).size > 0 and float(np.asarray(problem.w_cbf).reshape(-1)[0]) > 0.0:
            lhs = np.asarray(problem.A_cbf, dtype=np.float32) @ action
            rhs = np.asarray(problem.b_cbf, dtype=np.float32).reshape(-1)
            cbf_slack = np.maximum(lhs - rhs, 0.0).astype(np.float32)
            cbf_cap = float(np.asarray(problem.cbf_slack_max).reshape(-1)[0])
            if cbf_cap > 0.0:
                cbf_slack = np.minimum(cbf_slack, cbf_cap).astype(np.float32)
        objective = np.asarray([0.0], dtype=np.float32)
        return QPSolution(
            action=action,
            slack=slack,
            objective=objective,
            feasible=True,
            solver_status="stub_clipped",
            cbf_slack=cbf_slack,
        )

    def _build_layer(self, m_cbf: int, m_clf: int, n_u: int) -> CvxpyLayer:
        cache_key = (m_cbf, m_clf, n_u)
        if cache_key in self._cache:
            return self._cache[cache_key]

        u = cp.Variable(n_u)
        delta = cp.Variable(1, nonneg=True)
        eps_cbf = cp.Variable(m_cbf, nonneg=True) if m_cbf > 0 else None

        H_sqrt = cp.Parameter((n_u, n_u))
        f_lin = cp.Parameter(n_u)
        w_clf = cp.Parameter(1, nonneg=True)
        w_cbf = cp.Parameter(1, nonneg=True) if m_cbf > 0 else None
        A_cbf = cp.Parameter((m_cbf, n_u)) if m_cbf > 0 else None
        b_cbf = cp.Parameter(m_cbf) if m_cbf > 0 else None
        A_clf = cp.Parameter((m_clf, n_u)) if m_clf > 0 else None
        b_clf = cp.Parameter(m_clf) if m_clf > 0 else None
        u_min = cp.Parameter(n_u)
        u_max = cp.Parameter(n_u)
        cbf_slack_max = cp.Parameter(1, nonneg=True) if m_cbf > 0 else None

        objective = 0.5 * cp.sum_squares(H_sqrt @ u) + (f_lin @ u) + cp.sum(cp.multiply(w_clf, delta))
        if m_cbf > 0 and eps_cbf is not None and w_cbf is not None:
            objective += cp.sum(cp.multiply(w_cbf, eps_cbf))

        constraints = [u >= u_min, u <= u_max, delta >= 0]
        if m_cbf > 0 and eps_cbf is not None and A_cbf is not None and b_cbf is not None and cbf_slack_max is not None:
            constraints.append(A_cbf @ u <= b_cbf + eps_cbf)
            constraints.append(eps_cbf <= cp.multiply(np.ones((m_cbf,), dtype=np.float32), cbf_slack_max))
        if m_clf > 0 and A_clf is not None and b_clf is not None:
            constraints.append(A_clf @ u <= b_clf + delta)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        if not problem.is_dpp():
            raise RuntimeError("QP is not DPP-compliant for cvxpylayers")

        params = [H_sqrt, f_lin, w_clf, u_min, u_max]
        if m_cbf > 0:
            params.extend([w_cbf, cbf_slack_max, A_cbf, b_cbf])
        if m_clf > 0:
            params.extend([A_clf, b_clf])
        variables = [u, delta]
        if m_cbf > 0 and eps_cbf is not None:
            variables.append(eps_cbf)
        layer = CvxpyLayer(problem, parameters=params, variables=variables)
        self._cache[cache_key] = layer
        return layer

    def _solve_with_cvxpylayers(self, problem: QPProblem) -> QPSolution:
        assert torch is not None
        device = torch.device("cpu")

        A_cbf = np.asarray(problem.A_cbf, dtype=np.float32)
        b_cbf = np.asarray(problem.b_cbf, dtype=np.float32).reshape(-1)
        A_clf = np.asarray(problem.A_clf, dtype=np.float32)
        b_clf = np.asarray(problem.b_clf, dtype=np.float32).reshape(-1)
        H_mat = _symmetrize_spd_numpy(np.asarray(problem.H_mat, dtype=np.float32).reshape(self.action_dim, self.action_dim))
        H_sqrt = np.linalg.cholesky(H_mat).T.astype(np.float32)
        f_lin = np.asarray(problem.f_lin, dtype=np.float32).reshape(-1)
        n_u = int(f_lin.shape[0])

        layer = self._build_layer(A_cbf.shape[0], A_clf.shape[0], n_u)
        params = [
            torch.as_tensor(H_sqrt, device=device),
            torch.as_tensor(f_lin, device=device),
            torch.as_tensor(np.asarray(problem.w_clf, dtype=np.float32).reshape(-1), device=device),
            torch.as_tensor(np.asarray(problem.u_min, dtype=np.float32).reshape(-1), device=device),
            torch.as_tensor(np.asarray(problem.u_max, dtype=np.float32).reshape(-1), device=device),
        ]
        if A_cbf.shape[0] > 0:
            params.extend(
                [
                    torch.as_tensor(np.asarray(problem.w_cbf, dtype=np.float32).reshape(-1), device=device),
                    torch.as_tensor(np.asarray(problem.cbf_slack_max, dtype=np.float32).reshape(-1), device=device),
                    torch.as_tensor(A_cbf, device=device),
                    torch.as_tensor(b_cbf, device=device),
                ]
            )
        if A_clf.shape[0] > 0:
            params.extend(
                [
                    torch.as_tensor(A_clf, device=device),
                    torch.as_tensor(b_clf, device=device),
                ]
            )

        try:
            outputs = layer(
                *params,
                solver_args={
                    "solve_method": "SCS",
                    "max_iters": self.scs_max_iters,
                    "eps": self.scs_eps,
                },
            )
            status = "optimal_scs"
        except Exception:
            try:
                outputs = layer(
                    *params,
                    solver_args={
                        "solve_method": "ECOS",
                        "max_iters": self.ecos_max_iters,
                    },
                )
                status = "optimal"
            except Exception:
                return solve_reduced_fallback_qp(
                    problem,
                    action_dim=self.action_dim,
                    ecos_max_iters=self.ecos_max_iters,
                    scs_max_iters=self.scs_max_iters,
                    scs_eps=self.scs_eps,
                )

        u_sol, delta_sol = outputs[0], outputs[1]
        eps_cbf_sol = outputs[2] if A_cbf.shape[0] > 0 and len(outputs) > 2 else None
        action = u_sol.detach().cpu().numpy().astype(np.float32)
        slack = delta_sol.detach().cpu().numpy().astype(np.float32)
        cbf_slack = (
            eps_cbf_sol.detach().cpu().numpy().astype(np.float32)
            if eps_cbf_sol is not None
            else np.zeros((A_cbf.shape[0],), dtype=np.float32)
        )
        return QPSolution(
            action=action,
            slack=slack,
            objective=np.asarray([0.0], dtype=np.float32),
            feasible=True,
            solver_status=status,
            cbf_slack=cbf_slack,
        )
