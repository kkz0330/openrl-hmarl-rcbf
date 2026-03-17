import numpy as np
import pytest

from hmarl_cbf.skills import build_default_skill_library, validate_skill_spec
from hmarl_cbf.types import AgentObsLow, AgentState, LidarScan, SkillSpec


def test_default_skill_schema_valid() -> None:
    skills = build_default_skill_library(max_duration=12)
    assert len(skills) == 6
    for skill in skills:
        validate_skill_spec(skill)


def test_skill_schema_rejects_invalid_duration() -> None:
    with pytest.raises(ValueError):
        SkillSpec(
            skill_id=0,
            name="invalid",
            initiation_set_fn=lambda s, c: True,
            termination_set_fn=lambda s, c: False,
            max_duration=0,
            termination_fn=lambda s, c, t: False,
            safety_constraints_fn=lambda s, c: {},
            intrinsic_reward_fn=lambda s, a, c: 0.0,
            safe_skill_policy=lambda o, c: np.zeros(2, dtype=np.float32),
        )


def test_skill_schema_callable_contract() -> None:
    state = AgentState(
        agent_id=0,
        position=np.zeros(2, dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        goal=np.ones(2, dtype=np.float32),
    )
    obs = AgentObsLow(
        self_state=np.zeros(4, dtype=np.float32),
        goal_relative=np.ones(2, dtype=np.float32),
        lidar_scan=LidarScan(ranges=np.ones(8, dtype=np.float32), max_range=5.0),
        neighbor_summary=np.zeros(8, dtype=np.float32),
    )
    skill = build_default_skill_library(max_duration=10)[0]
    assert isinstance(skill.initiation_set_fn(state, {}), bool)
    assert isinstance(skill.termination_fn(state, {}, 1), bool)
    action = skill.safe_skill_policy(obs, {})
    assert action.shape == (2,)
