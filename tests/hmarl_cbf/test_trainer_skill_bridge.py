import numpy as np

from hmarl_cbf.buffer import HierRolloutBuffer
from hmarl_cbf.control import ConstraintBuilder, DifferentiableQPSolver, LowLevelSafeController, SyncCoordinator
from hmarl_cbf.skills import SKILL_ACCELERATE, SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.train import TrainerSyncOnPolicy
from hmarl_cbf.types import AgentObsLow, AgentState, LidarScan, QPParam


class DummyLowPolicy:
    def __call__(self, obs_low: AgentObsLow, skill_id: int) -> QPParam:
        del obs_low, skill_id
        return QPParam(
            u_ref=np.array([0.0, 0.0], dtype=np.float32),
            r_diag=np.array([1.0, 1.0], dtype=np.float32),
            w_clf=np.array([1.0], dtype=np.float32),
            cbf_k0=np.array([1.0], dtype=np.float32),
            cbf_k1=np.array([1.0], dtype=np.float32),
            clf_k=np.array([1.0], dtype=np.float32),
        )


def _state(agent_id: int, pos, vel, goal) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        position=np.asarray(pos, dtype=np.float32),
        velocity=np.asarray(vel, dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
    )


def _obs(state: AgentState) -> AgentObsLow:
    return AgentObsLow(
        self_state=np.concatenate([state.position, state.velocity], axis=0).astype(np.float32),
        goal_relative=(state.goal - state.position).astype(np.float32),
        lidar_scan=LidarScan(ranges=np.ones(8, dtype=np.float32), max_range=5.0),
        neighbor_summary=np.zeros(8, dtype=np.float32),
    )


def test_trainer_skill_runtime_bridge_outputs_beta_flags() -> None:
    skills = build_default_skill_library(max_duration=2)
    runtime = SkillRuntimeManager(skills, default_ctx={"goal_threshold": 0.1})

    trainer = TrainerSyncOnPolicy(
        env=object(),
        high_policy=object(),
        low_policy=object(),
        constraint_builder=ConstraintBuilder(),
        qp_solver=DifferentiableQPSolver(use_stub_if_unavailable=True),
        coordinator=SyncCoordinator(num_agents=2, t_sync_max=5),
        buffer=HierRolloutBuffer(),
        skill_runtime=runtime,
    )

    states = {
        0: _state(0, [0.0, 0.0], [0.0, 0.0], [10.0, 0.0]),
        1: _state(1, [1.0, 0.0], [0.0, 0.0], [10.0, 0.0]),
    }
    obs_low = {0: _obs(states[0]), 1: _obs(states[1])}
    trainer.activate_round_skills({0: SKILL_ACCELERATE, 1: SKILL_ACCELERATE}, states, extra_ctx={"target_speed": 10.0})

    beta1 = trainer.evaluate_skill_step(states=states, obs_low=obs_low, actions={0: np.zeros(2), 1: np.zeros(2)})
    beta2 = trainer.evaluate_skill_step(states=states, obs_low=obs_low, actions={0: np.zeros(2), 1: np.zeros(2)})

    assert beta1[0] is False
    assert beta2[0] is True


def test_trainer_compute_safe_actions_bridge() -> None:
    skills = build_default_skill_library(max_duration=3)
    runtime = SkillRuntimeManager(skills, default_ctx={"goal_threshold": 0.1})
    constraint_builder = ConstraintBuilder(u_min=[-1.0, -1.0], u_max=[1.0, 1.0])
    low_controller = LowLevelSafeController(
        low_policy=DummyLowPolicy(),
        constraint_builder=constraint_builder,
        qp_solver=DifferentiableQPSolver(use_stub_if_unavailable=True),
        skill_ref_weight=1.0,
    )
    trainer = TrainerSyncOnPolicy(
        env=object(),
        high_policy=object(),
        low_policy=object(),
        constraint_builder=constraint_builder,
        qp_solver=DifferentiableQPSolver(use_stub_if_unavailable=True),
        coordinator=SyncCoordinator(num_agents=2, t_sync_max=5),
        buffer=HierRolloutBuffer(),
        skill_runtime=runtime,
        low_level_controller=low_controller,
    )
    states = {
        0: _state(0, [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]),
        1: _state(1, [1.5, 0.0], [0.0, 0.0], [2.0, 0.0]),
    }
    obs_low = {0: _obs(states[0]), 1: _obs(states[1])}
    trainer.activate_round_skills({0: SKILL_ACCELERATE, 1: SKILL_ACCELERATE}, states)
    actions, outputs = trainer.compute_safe_actions(states=states, obs_low=obs_low, obstacles=[])
    assert set(actions.keys()) == {0, 1}
    assert set(outputs.keys()) == {0, 1}
    assert np.asarray(actions[0]).shape == (2,)
