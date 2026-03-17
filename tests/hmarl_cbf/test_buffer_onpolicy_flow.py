import numpy as np

from hmarl_cbf.buffer import HierRolloutBuffer
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


def test_high_option_open_close_and_gae() -> None:
    buf = HierRolloutBuffer()
    buf.start_high_option(k=0, agent_id=0, t_start=0, obs_high=_obs_high(), skill_id=1, logp=-0.1, value=0.3)
    buf.close_high_option(agent_id=0, t_end=3, return_ext=1.0, done=False, sync_switch=True)
    buf.start_high_option(k=1, agent_id=0, t_start=3, obs_high=_obs_high(), skill_id=2, logp=-0.2, value=0.2)
    buf.close_high_option(agent_id=0, t_end=5, return_ext=0.5, done=True, sync_switch=True)

    stats = buf.compute_high_advantages(gamma=1.0, lam=1.0, use_gae=True)
    assert stats["n_samples"] == 2.0
    seq = buf.high_by_agent()[0]
    # Option-1 target = 0.5, advantage = 0.3
    assert abs(float(seq[1].advantage) - 0.3) < 1e-5
    assert abs(float(seq[1].value_target) - 0.5) < 1e-5
    # Option-0 target = 1.5, advantage = 1.2
    assert abs(float(seq[0].advantage) - 1.2) < 1e-5
    assert abs(float(seq[0].value_target) - 1.5) < 1e-5


def test_low_returns_reset_on_sync_switch() -> None:
    buf = HierRolloutBuffer()
    obs = _obs_low()
    buf.add_low_step(
        LowStepTransition(
            t=0,
            agent_id=0,
            obs_low=obs,
            skill_id=0,
            action=np.zeros(2, dtype=np.float32),
            reward_int=1.0,
            reward_ext=0.0,
            done=False,
            sync_switch=False,
        )
    )
    buf.add_low_step(
        LowStepTransition(
            t=1,
            agent_id=0,
            obs_low=obs,
            skill_id=0,
            action=np.zeros(2, dtype=np.float32),
            reward_int=2.0,
            reward_ext=0.0,
            done=False,
            sync_switch=True,
        )
    )
    buf.add_low_step(
        LowStepTransition(
            t=2,
            agent_id=0,
            obs_low=obs,
            skill_id=0,
            action=np.zeros(2, dtype=np.float32),
            reward_int=3.0,
            reward_ext=0.0,
            done=False,
            sync_switch=False,
        )
    )
    buf.add_low_step(
        LowStepTransition(
            t=3,
            agent_id=0,
            obs_low=obs,
            skill_id=0,
            action=np.zeros(2, dtype=np.float32),
            reward_int=4.0,
            reward_ext=0.0,
            done=True,
            sync_switch=False,
        )
    )
    stats = buf.compute_low_returns(gamma=1.0, ext_reward_coef=0.0, reset_on_sync_switch=True)
    assert stats["n_samples"] == 4.0
    seq = buf.low_by_agent()[0]
    assert abs(float(seq[0].return_target) - 3.0) < 1e-5
    assert abs(float(seq[1].return_target) - 2.0) < 1e-5
    assert abs(float(seq[2].return_target) - 7.0) < 1e-5
    assert abs(float(seq[3].return_target) - 4.0) < 1e-5
