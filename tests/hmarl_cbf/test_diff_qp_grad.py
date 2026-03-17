import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cvxpy")
pytest.importorskip("cvxpylayers")

from hmarl_cbf.control import TorchDifferentiableQPSolver, build_diff_constraint_constants
from hmarl_cbf.policies import LowLevelQPPolicy
from hmarl_cbf.train.gradient_check_diff_qp import run_gradient_check
from hmarl_cbf.types import AgentState, QPParam


def _qp_param_from_policy(policy: LowLevelQPPolicy, obs: torch.Tensor, skill_id: int) -> QPParam:
    out = policy(obs.unsqueeze(0), torch.tensor([skill_id], dtype=torch.long))
    return QPParam(
        u_ref=out.u_ref.reshape(-1),
        r_diag=out.r_diag.reshape(-1),
        w_clf=out.w_clf.reshape(-1),
        cbf_k0=out.cbf_k0.reshape(-1),
        cbf_k1=out.cbf_k1.reshape(-1),
        clf_k=out.clf_k.reshape(-1),
    )


def test_diff_qp_forward_shapes() -> None:
    torch.manual_seed(1)
    policy = LowLevelQPPolicy(obs_dim=18, n_skills=6, action_dim=2, hidden_dim=32)
    solver = TorchDifferentiableQPSolver(action_dim=2)

    state_i = AgentState(
        agent_id=0,
        position=np.array([0.0, 0.0], dtype=np.float32),
        velocity=np.array([0.3, 0.0], dtype=np.float32),
        goal=np.array([2.0, 0.0], dtype=np.float32),
    )
    state_j = AgentState(
        agent_id=1,
        position=np.array([1.0, 0.0], dtype=np.float32),
        velocity=np.array([0.0, 0.0], dtype=np.float32),
        goal=np.array([2.0, 0.0], dtype=np.float32),
    )
    constants = build_diff_constraint_constants(
        state_i=state_i,
        neighbors=[state_j],
        obstacles=[{"center": np.array([2.0, 1.0], dtype=np.float32), "radius": 0.4}],
    )
    obs = torch.randn(18, dtype=torch.float32)
    qp_param = _qp_param_from_policy(policy, obs, skill_id=2)
    result = solver.solve(qp_param, constants)

    assert result.action.shape == (2,)
    assert result.slack.shape == (1,)
    assert torch.isfinite(result.action).all()


def test_diff_qp_gradient_check() -> None:
    result = run_gradient_check(eps=1e-3, tol=1e-1)
    assert result.rel_error <= 1e-1
