import numpy as np

from hmarl_cbf.control import ConstraintBuilder, DifferentiableQPSolver, LowLevelSafeController
from hmarl_cbf.types import AgentObsLow, AgentState, LidarScan, QPParam


class DummyLowPolicy:
    def __call__(self, obs_low: AgentObsLow, skill_id: int) -> QPParam:
        del obs_low, skill_id
        return QPParam(
            u_ref=np.array([0.2, 0.1], dtype=np.float32),
            r_diag=np.array([1.0, 1.0], dtype=np.float32),
            w_clf=np.array([1.0], dtype=np.float32),
            cbf_k0=np.array([1.0], dtype=np.float32),
            cbf_k1=np.array([1.0], dtype=np.float32),
            clf_k=np.array([1.0], dtype=np.float32),
        )


def _state(agent_id: int, p, v, g) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        position=np.asarray(p, dtype=np.float32),
        velocity=np.asarray(v, dtype=np.float32),
        goal=np.asarray(g, dtype=np.float32),
    )


def _obs(state: AgentState) -> AgentObsLow:
    return AgentObsLow(
        self_state=np.concatenate([state.position, state.velocity], axis=0).astype(np.float32),
        goal_relative=(state.goal - state.position).astype(np.float32),
        lidar_scan=LidarScan(ranges=np.ones(8, dtype=np.float32), max_range=5.0),
        neighbor_summary=np.zeros(8, dtype=np.float32),
    )


def test_low_level_controller_fuses_skill_reference() -> None:
    controller = LowLevelSafeController(
        low_policy=DummyLowPolicy(),
        constraint_builder=ConstraintBuilder(u_min=[-1.0, -1.0], u_max=[1.0, 1.0]),
        qp_solver=DifferentiableQPSolver(use_stub_if_unavailable=True),
        skill_ref_weight=0.75,
    )
    states = {
        0: _state(0, [0.0, 0.0], [0.0, 0.0], [2.0, 0.0]),
        1: _state(1, [1.5, 0.0], [0.0, 0.0], [2.0, 1.0]),
    }
    obs_low = {0: _obs(states[0]), 1: _obs(states[1])}
    skill_targets = {
        0: {"skill_id": 2, "u_ref_skill": np.array([1.0, 0.0], dtype=np.float32), "safety_constraints": {}},
        1: {"skill_id": 4, "u_ref_skill": np.array([0.0, 1.0], dtype=np.float32), "safety_constraints": {}},
    }
    actions, outputs = controller.solve_batch(states=states, obs_low=obs_low, skill_targets=skill_targets, obstacles=[])
    assert set(actions.keys()) == {0, 1}
    expected_fused_0 = 0.75 * np.array([1.0, 0.0], dtype=np.float32) + 0.25 * np.array([0.2, 0.1], dtype=np.float32)
    assert np.allclose(outputs[0].fused_u_ref, expected_fused_0, atol=1e-6)
    assert np.all(actions[0] <= 1.0) and np.all(actions[0] >= -1.0)


def test_low_level_controller_builds_cbf_only_for_perceived_entities() -> None:
    controller = LowLevelSafeController(
        low_policy=DummyLowPolicy(),
        constraint_builder=ConstraintBuilder(u_min=[-1.0, -1.0], u_max=[1.0, 1.0]),
        qp_solver=DifferentiableQPSolver(use_stub_if_unavailable=True),
        skill_ref_weight=0.7,
        neighbor_perception_radius=1.0,
        obstacle_perception_range=1.0,
    )
    states = {
        0: _state(0, [0.0, 0.0], [0.0, 0.0], [2.0, 0.0]),
        1: _state(1, [0.6, 0.0], [0.0, 0.0], [2.0, 1.0]),  # perceived neighbor
        2: _state(2, [2.5, 0.0], [0.0, 0.0], [2.0, -1.0]),  # out-of-range neighbor
    }
    obs_low = {aid: _obs(st) for aid, st in states.items()}
    skill_targets = {
        0: {"skill_id": 2, "u_ref_skill": np.array([0.0, 0.0], dtype=np.float32), "safety_constraints": {}},
        1: {"skill_id": 2, "u_ref_skill": np.array([0.0, 0.0], dtype=np.float32), "safety_constraints": {}},
        2: {"skill_id": 2, "u_ref_skill": np.array([0.0, 0.0], dtype=np.float32), "safety_constraints": {}},
    }
    obstacles = [
        {"center": np.array([0.8, 0.0], dtype=np.float32), "radius": 0.2},  # perceived
        {"center": np.array([3.0, 0.0], dtype=np.float32), "radius": 0.2},  # out-of-range
    ]

    _, outputs = controller.solve_batch(states=states, obs_low=obs_low, skill_targets=skill_targets, obstacles=obstacles)
    # For agent 0: one perceived neighbor + one perceived obstacle => 2 CBF rows.
    assert np.asarray(outputs[0].problem.A_cbf).shape[0] == 2
