import numpy as np

from hmarl_cbf.buffer import HierRolloutBuffer
from hmarl_cbf.control import ConstraintBuilder, DifferentiableQPSolver, SyncCoordinator
from hmarl_cbf.train import TrainerSyncOnPolicy
from hmarl_cbf.types import AgentObsHigh, AgentObsLow, LidarScan, LowStepTransition


def _obs_high() -> AgentObsHigh:
    return AgentObsHigh(
        self_state=np.zeros(4, dtype=np.float32),
        goal_relative=np.zeros(2, dtype=np.float32),
        neighbor_summary=np.zeros(8, dtype=np.float32),
    )


def _obs_low() -> AgentObsLow:
    return AgentObsLow(
        self_state=np.zeros(4, dtype=np.float32),
        goal_relative=np.zeros(2, dtype=np.float32),
        lidar_scan=LidarScan(ranges=np.ones(8, dtype=np.float32), max_range=5.0),
        neighbor_summary=np.zeros(8, dtype=np.float32),
    )


def test_trainer_finalize_rollout_buffers() -> None:
    buffer = HierRolloutBuffer()
    trainer = TrainerSyncOnPolicy(
        env=object(),
        high_policy=object(),
        low_policy=object(),
        constraint_builder=ConstraintBuilder(),
        qp_solver=DifferentiableQPSolver(use_stub_if_unavailable=True),
        coordinator=SyncCoordinator(num_agents=1, t_sync_max=10),
        buffer=buffer,
    )

    buffer.start_high_option(k=0, agent_id=0, t_start=0, obs_high=_obs_high(), skill_id=1, logp=-0.1, value=0.1)
    buffer.close_high_option(agent_id=0, t_end=2, return_ext=0.8, done=False, sync_switch=True)
    buffer.start_high_option(k=1, agent_id=0, t_start=2, obs_high=_obs_high(), skill_id=2, logp=-0.2, value=0.2)
    buffer.close_high_option(agent_id=0, t_end=4, return_ext=0.3, done=True, sync_switch=True)

    buffer.add_low_step(
        LowStepTransition(
            t=0,
            agent_id=0,
            obs_low=_obs_low(),
            skill_id=1,
            action=np.zeros(2, dtype=np.float32),
            reward_int=0.1,
            reward_ext=0.2,
            done=False,
            sync_switch=False,
        )
    )
    buffer.add_low_step(
        LowStepTransition(
            t=1,
            agent_id=0,
            obs_low=_obs_low(),
            skill_id=1,
            action=np.zeros(2, dtype=np.float32),
            reward_int=0.1,
            reward_ext=0.2,
            done=False,
            sync_switch=True,
        )
    )

    stats = trainer.finalize_rollout_buffers(
        gamma_high=0.99,
        lam_high=0.95,
        gamma_low=0.99,
        low_ext_reward_coef=0.5,
    )
    assert stats["high_n"] == 2.0
    assert stats["low_n"] == 2.0
    assert np.isfinite(stats["high_adv_mean"])
    assert np.isfinite(stats["low_return_mean"])
