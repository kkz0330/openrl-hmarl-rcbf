from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime script
    raise RuntimeError("PyTorch is required to run gradient check") from exc

from hmarl_cbf.control import TorchDifferentiableQPSolver, build_diff_constraint_constants
from hmarl_cbf.policies import LowLevelQPPolicy
from hmarl_cbf.types import AgentState, QPParam


@dataclass(slots=True)
class GradCheckResult:
    autograd: float
    finite_diff: float
    rel_error: float


def _build_qp_param(policy: LowLevelQPPolicy, obs: torch.Tensor, skill_id: int) -> QPParam:
    qp = policy(obs.unsqueeze(0), torch.tensor([skill_id], dtype=torch.long))
    return QPParam(
        u_ref=qp.u_ref.reshape(-1),
        r_diag=qp.r_diag.reshape(-1),
        w_clf=qp.w_clf.reshape(-1),
        cbf_k0=qp.cbf_k0.reshape(-1),
        cbf_k1=qp.cbf_k1.reshape(-1),
        clf_k=qp.clf_k.reshape(-1),
        f_lin=None,
        hocbf_gamma_h=None,
        hocbf_gamma_hdot=None,
    )


def run_gradient_check(eps: float = 1e-3, tol: float = 5e-2) -> GradCheckResult:
    torch.manual_seed(0)
    np.random.seed(0)

    obs_dim = 18
    policy = LowLevelQPPolicy(obs_dim=obs_dim, n_skills=6, action_dim=2, hidden_dim=64)
    solver = TorchDifferentiableQPSolver(action_dim=2)

    state_i = AgentState(
        agent_id=0,
        position=np.array([0.0, 0.0], dtype=np.float32),
        velocity=np.array([0.2, 0.1], dtype=np.float32),
        goal=np.array([3.0, 0.0], dtype=np.float32),
    )
    neighbor = AgentState(
        agent_id=1,
        position=np.array([1.5, 0.2], dtype=np.float32),
        velocity=np.array([0.0, 0.0], dtype=np.float32),
        goal=np.array([3.0, 0.0], dtype=np.float32),
    )
    obstacles = [{"center": np.array([2.0, 1.0], dtype=np.float32), "radius": 0.5}]

    constants = build_diff_constraint_constants(
        state_i=state_i,
        neighbors=[neighbor],
        obstacles=obstacles,
        d_min_agent=0.6,
        d_safe_obs=0.6,
        u_min=(-1.0, -1.0),
        u_max=(1.0, 1.0),
    )
    obs = torch.randn(obs_dim, dtype=torch.float32)
    skill_id = 2
    target = torch.tensor([0.25, -0.1], dtype=torch.float32)

    # Pick one scalar weight for finite-diff check.
    param = policy.r_diag_head.weight
    idx = (0, 0)
    original = float(param.data[idx].item())

    policy.zero_grad(set_to_none=True)
    qp_param = _build_qp_param(policy, obs, skill_id)
    out = solver.solve(qp_param, constants)
    loss = 0.5 * torch.sum((out.action - target) ** 2)
    loss.backward()
    autograd = float(param.grad[idx].item())

    with torch.no_grad():
        param.data[idx] = original + eps
    qp_param_p = _build_qp_param(policy, obs, skill_id)
    out_p = solver.solve(qp_param_p, constants)
    loss_p = 0.5 * torch.sum((out_p.action - target) ** 2)

    with torch.no_grad():
        param.data[idx] = original - eps
    qp_param_m = _build_qp_param(policy, obs, skill_id)
    out_m = solver.solve(qp_param_m, constants)
    loss_m = 0.5 * torch.sum((out_m.action - target) ** 2)

    with torch.no_grad():
        param.data[idx] = original

    finite_diff = float(((loss_p - loss_m) / (2.0 * eps)).item())
    denom = max(1.0, abs(autograd), abs(finite_diff))
    rel_error = abs(autograd - finite_diff) / denom
    if rel_error > tol:
        raise AssertionError(
            f"Gradient check failed: autograd={autograd:.6f}, finite_diff={finite_diff:.6f}, rel_error={rel_error:.6f}, tol={tol:.6f}"
        )
    return GradCheckResult(autograd=autograd, finite_diff=finite_diff, rel_error=rel_error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-difference gradient check for differentiable QP path.")
    parser.add_argument("--eps", type=float, default=1e-3, help="Finite difference epsilon.")
    parser.add_argument("--tol", type=float, default=5e-2, help="Relative error tolerance.")
    args = parser.parse_args()

    result = run_gradient_check(eps=args.eps, tol=args.tol)
    print(
        "gradient_check_passed "
        f"autograd={result.autograd:.6f} finite_diff={result.finite_diff:.6f} rel_error={result.rel_error:.6f}"
    )


if __name__ == "__main__":
    main()
