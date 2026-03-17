import numpy as np

from hmarl_cbf.control import ConstraintBuilder, DifferentiableQPSolver
from hmarl_cbf.types import AgentState, QPParam


def test_qp_io_contract() -> None:
    state_i = AgentState(
        agent_id=0,
        position=np.array([0.0, 0.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        goal=np.array([1.0, 0.0], dtype=np.float32),
    )
    state_j = AgentState(
        agent_id=1,
        position=np.array([1.0, 0.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        goal=np.array([0.0, 0.0], dtype=np.float32),
    )
    qp_param = QPParam(
        u_ref=np.array([0.2, 0.0], dtype=np.float32),
        r_diag=np.array([1.0, 1.0], dtype=np.float32),
        w_clf=np.array([1.0], dtype=np.float32),
        cbf_k0=np.array([1.0], dtype=np.float32),
        cbf_k1=np.array([1.0], dtype=np.float32),
        clf_k=np.array([1.0], dtype=np.float32),
    )
    builder = ConstraintBuilder(d_min_agent=0.5, d_safe_obs=0.5)
    problem = builder.build_for_agent(
        state_i=state_i,
        neighbors=[state_j],
        obstacles=[{"center": np.array([2.0, 2.0], dtype=np.float32), "radius": 0.4}],
        qp_param=qp_param,
    )
    solver = DifferentiableQPSolver(action_dim=2, use_stub_if_unavailable=True)
    solution = solver.solve(problem)

    assert np.asarray(problem.A_cbf).shape[1] == 2
    assert np.asarray(problem.A_clf).shape[1] == 2
    assert np.asarray(solution.action).shape == (2,)
    assert np.asarray(solution.slack).shape == (1,)
