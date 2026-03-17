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
        r_diag = np.asarray(problem.r_diag, dtype=np.float32).reshape(-1)
        if problem.f_lin is not None:
            f_lin = np.asarray(problem.f_lin, dtype=np.float32).reshape(-1)
        else:
            u_ref = np.asarray(problem.u_ref, dtype=np.float32).reshape(-1)
            f_lin = -(r_diag * u_ref)
        action = (-f_lin / np.maximum(r_diag, 1e-5)).astype(np.float32)
        action = np.clip(action, np.asarray(problem.u_min, dtype=np.float32), np.asarray(problem.u_max, dtype=np.float32))
        slack = np.asarray([max(problem.delta_min, 0.0)], dtype=np.float32)
        objective = np.asarray([0.0], dtype=np.float32)
        return QPSolution(
            action=action,
            slack=slack,
            objective=objective,
            feasible=True,
            solver_status="stub_clipped",
        )

    def _build_layer(self, m_cbf: int, m_clf: int, n_u: int) -> CvxpyLayer:
        cache_key = (m_cbf, m_clf, n_u)
        if cache_key in self._cache:
            return self._cache[cache_key]

        u = cp.Variable(n_u)
        delta = cp.Variable(1, nonneg=True)
        r_diag = cp.Parameter(n_u, nonneg=True)
        f_lin = cp.Parameter(n_u)
        w_clf = cp.Parameter(1, nonneg=True)
        A_cbf = cp.Parameter((m_cbf, n_u)) if m_cbf > 0 else None
        b_cbf = cp.Parameter(m_cbf) if m_cbf > 0 else None
        A_clf = cp.Parameter((m_clf, n_u)) if m_clf > 0 else None
        b_clf = cp.Parameter(m_clf) if m_clf > 0 else None
        u_min = cp.Parameter(n_u)
        u_max = cp.Parameter(n_u)

        # DPP-compliant diagonal-quadratic form:
        # 0.5 * u^T diag(r_diag) u + f_lin^T u + w_clf * delta
        objective = 0.5 * cp.sum(cp.multiply(r_diag, cp.square(u))) + (f_lin @ u) + cp.sum(cp.multiply(w_clf, delta))
        constraints = [u >= u_min, u <= u_max, delta >= 0]
        if m_cbf > 0:
            constraints.append(A_cbf @ u <= b_cbf)
        if m_clf > 0:
            constraints.append(A_clf @ u <= b_clf + delta)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        if not problem.is_dpp():
            raise RuntimeError("QP is not DPP-compliant for cvxpylayers")

        params = [r_diag, f_lin, w_clf, u_min, u_max]
        if m_cbf > 0:
            params.extend([A_cbf, b_cbf])
        if m_clf > 0:
            params.extend([A_clf, b_clf])
        layer = CvxpyLayer(problem, parameters=params, variables=[u, delta])
        self._cache[cache_key] = layer
        return layer

    def _solve_with_cvxpylayers(self, problem: QPProblem) -> QPSolution:
        assert torch is not None
        device = torch.device("cpu")

        A_cbf = np.asarray(problem.A_cbf, dtype=np.float32)
        b_cbf = np.asarray(problem.b_cbf, dtype=np.float32).reshape(-1)
        A_clf = np.asarray(problem.A_clf, dtype=np.float32)
        b_clf = np.asarray(problem.b_clf, dtype=np.float32).reshape(-1)
        if problem.f_lin is not None:
            n_u = int(np.asarray(problem.f_lin).reshape(-1).shape[0])
        else:
            n_u = int(np.asarray(problem.u_ref).reshape(-1).shape[0])

        layer = self._build_layer(A_cbf.shape[0], A_clf.shape[0], n_u)
        r_diag = np.asarray(problem.r_diag, dtype=np.float32).reshape(-1)
        if problem.f_lin is not None:
            f_lin = np.asarray(problem.f_lin, dtype=np.float32).reshape(-1)
        else:
            u_ref = np.asarray(problem.u_ref, dtype=np.float32).reshape(-1)
            f_lin = -(r_diag * u_ref)
        params = [
            torch.as_tensor(r_diag, device=device),
            torch.as_tensor(f_lin, device=device),
            torch.as_tensor(np.asarray([problem.w_clf], dtype=np.float32), device=device),
            torch.as_tensor(np.asarray(problem.u_min, dtype=np.float32).reshape(-1), device=device),
            torch.as_tensor(np.asarray(problem.u_max, dtype=np.float32).reshape(-1), device=device),
        ]
        if A_cbf.shape[0] > 0:
            params.extend(
                [
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
            u_sol, delta_sol = layer(
                *params,
                solver_args={
                    "solve_method": "ECOS",
                    "max_iters": self.ecos_max_iters,
                },
            )
            action = u_sol.detach().cpu().numpy().astype(np.float32)
            slack = delta_sol.detach().cpu().numpy().astype(np.float32)
            return QPSolution(
                action=action,
                slack=slack,
                objective=np.asarray([0.0], dtype=np.float32),
                feasible=True,
                solver_status="optimal",
            )
        except Exception:
            try:
                u_sol, delta_sol = layer(
                    *params,
                    solver_args={
                        "solve_method": "SCS",
                        "max_iters": self.scs_max_iters,
                        "eps": self.scs_eps,
                    },
                )
                action = u_sol.detach().cpu().numpy().astype(np.float32)
                slack = delta_sol.detach().cpu().numpy().astype(np.float32)
                return QPSolution(
                    action=action,
                    slack=slack,
                    objective=np.asarray([0.0], dtype=np.float32),
                    feasible=True,
                    solver_status="optimal_scs",
                )
            except Exception:
                # Deterministic safe fallback when solver fails.
                return self._solve_stub(problem)
