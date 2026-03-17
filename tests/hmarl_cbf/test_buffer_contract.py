import numpy as np

from hmarl_cbf.buffer import HierRolloutBuffer
from hmarl_cbf.types import AgentObsHigh, AgentObsLow, HighOptionTransition, LidarScan, LowStepTransition


def _dummy_obs_high() -> AgentObsHigh:
    return AgentObsHigh(
        self_state=np.zeros(4, dtype=np.float32),
        goal_relative=np.ones(2, dtype=np.float32),
        neighbor_summary=np.zeros(8, dtype=np.float32),
    )


def _dummy_obs_low() -> AgentObsLow:
    return AgentObsLow(
        self_state=np.zeros(4, dtype=np.float32),
        goal_relative=np.ones(2, dtype=np.float32),
        lidar_scan=LidarScan(ranges=np.ones(8, dtype=np.float32), max_range=5.0),
        neighbor_summary=np.zeros(8, dtype=np.float32),
    )


def test_buffer_contract() -> None:
    buffer = HierRolloutBuffer()
    buffer.add_low_step(
        LowStepTransition(
            t=0,
            agent_id=0,
            obs_low=_dummy_obs_low(),
            skill_id=1,
            action=np.zeros(2, dtype=np.float32),
            reward_int=0.0,
            reward_ext=0.1,
            done=False,
        )
    )
    buffer.add_high_option(
        HighOptionTransition(
            k=0,
            agent_id=0,
            t_start=0,
            t_end=5,
            obs_high=_dummy_obs_high(),
            skill_id=1,
            logp=-0.5,
            value=0.2,
            return_ext=1.0,
            done=False,
        )
    )
    assert buffer.size_low() == 1
    assert buffer.size_high() == 1
    low, high = buffer.snapshot()
    assert len(low) == 1
    assert len(high) == 1
