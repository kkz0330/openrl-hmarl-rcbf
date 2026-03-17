import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cvxpy")
pytest.importorskip("cvxpylayers")

from hmarl_cbf.buffer import HierRolloutBuffer
from hmarl_cbf.control import ConstraintBuilder, DifferentiableQPSolver, SyncCoordinator, TorchDifferentiableQPSolver
from hmarl_cbf.policies import LowLevelQPPolicy
from hmarl_cbf.train import TrainerSyncOnPolicy
from hmarl_cbf.types import AgentObsLow, AgentState, LidarScan


def _state() -> AgentState:
    return AgentState(
        agent_id=0,
        position=np.array([0.0, 0.0], dtype=np.float32),
        velocity=np.array([0.2, 0.0], dtype=np.float32),
        goal=np.array([2.0, 0.0], dtype=np.float32),
    )


def _obs(state: AgentState) -> AgentObsLow:
    return AgentObsLow(
        self_state=np.concatenate([state.position, state.velocity], axis=0).astype(np.float32),
        goal_relative=(state.goal - state.position).astype(np.float32),
        lidar_scan=LidarScan(ranges=np.ones(8, dtype=np.float32), max_range=5.0),
        neighbor_summary=np.zeros(8, dtype=np.float32),
    )


def test_trainer_backward_low_level_diff_step_runs() -> None:
    low_policy = LowLevelQPPolicy(obs_dim=22, n_skills=6, action_dim=2, hidden_dim=32)
    trainer = TrainerSyncOnPolicy(
        env=object(),
        high_policy=object(),
        low_policy=low_policy,
        constraint_builder=ConstraintBuilder(),
        qp_solver=DifferentiableQPSolver(use_stub_if_unavailable=True),
        coordinator=SyncCoordinator(num_agents=1, t_sync_max=10),
        buffer=HierRolloutBuffer(),
        diff_qp_solver=TorchDifferentiableQPSolver(action_dim=2),
    )

    state_i = _state()
    obs_low = _obs(state_i)
    target_action = np.array([0.1, 0.0], dtype=np.float32)
    optimizer = torch.optim.Adam(low_policy.parameters(), lr=1e-3)

    before = low_policy.u_ref_head.weight.detach().clone()
    metrics = trainer.backward_low_level_diff_step(
        state_i=state_i,
        neighbors=[],
        obstacles=[],
        obs_low=obs_low,
        skill_id=2,
        target_action=target_action,
        optimizer=optimizer,
    )
    after = low_policy.u_ref_head.weight.detach().clone()

    assert "loss" in metrics and np.isfinite(metrics["loss"])
    assert not torch.equal(before, after)
