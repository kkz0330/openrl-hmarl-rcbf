import numpy as np
import pytest

from hmarl_cbf.skills import (
    SKILL_ACCELERATE,
    SKILL_CRUISE,
    SKILL_DECELERATE,
    SKILL_TURN_LEFT,
    build_default_skill_library,
)
from hmarl_cbf.skills.runtime import SkillRuntimeManager
from hmarl_cbf.types import AgentObsLow, AgentState, LidarScan


def _make_state(position, velocity, goal, agent_id: int = 0) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        position=np.asarray(position, dtype=np.float32),
        velocity=np.asarray(velocity, dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
    )


def _make_obs(state: AgentState) -> AgentObsLow:
    return AgentObsLow(
        self_state=np.concatenate([state.position, state.velocity], axis=0).astype(np.float32),
        goal_relative=(state.goal - state.position).astype(np.float32),
        lidar_scan=LidarScan(ranges=np.ones(16, dtype=np.float32), max_range=6.0),
        neighbor_summary=np.zeros(8, dtype=np.float32),
    )


def test_skill_runtime_timeout_termination() -> None:
    skills = build_default_skill_library(max_duration=3)
    mgr = SkillRuntimeManager(skills, default_ctx={"goal_threshold": 0.1})
    state = _make_state(position=[0.0, 0.0], velocity=[0.0, 0.0], goal=[10.0, 0.0], agent_id=0)
    obs = _make_obs(state)
    mgr.activate_skill(agent_id=0, skill_id=SKILL_ACCELERATE, state=state, extra_ctx={"target_speed": 10.0})

    out1 = mgr.step(agent_id=0, state=state, obs_low=obs, executed_action=np.zeros(2, dtype=np.float32))
    out2 = mgr.step(agent_id=0, state=state, obs_low=obs, executed_action=np.zeros(2, dtype=np.float32))
    out3 = mgr.step(agent_id=0, state=state, obs_low=obs, executed_action=np.zeros(2, dtype=np.float32))

    assert out1.beta is False
    assert out2.beta is False
    assert out3.beta is True
    assert out3.tau == 3


def test_turn_left_reference_action_direction() -> None:
    skills = build_default_skill_library(max_duration=10)
    mgr = SkillRuntimeManager(skills, default_ctx={"turn_target_angle": 0.5, "action_limit": 1.0})
    state = _make_state(position=[0.0, 0.0], velocity=[1.0, 0.0], goal=[5.0, 0.0], agent_id=0)
    obs = _make_obs(state)
    mgr.activate_skill(agent_id=0, skill_id=SKILL_TURN_LEFT, state=state)
    out = mgr.step(agent_id=0, state=state, obs_low=obs, executed_action=np.zeros(2, dtype=np.float32))

    # Heading along +x, left turn should yield positive y reference component.
    assert out.u_ref_skill.shape == (2,)
    assert out.u_ref_skill[1] > 0.0
    assert np.isfinite(out.intrinsic_reward)


def test_turn_initiation_requires_min_speed() -> None:
    skills = build_default_skill_library(max_duration=10)
    mgr = SkillRuntimeManager(skills, default_ctx={"turn_min_speed": 0.3})
    state = _make_state(position=[0.0, 0.0], velocity=[0.0, 0.0], goal=[1.0, 0.0], agent_id=0)
    with pytest.raises(ValueError):
        mgr.activate_skill(agent_id=0, skill_id=SKILL_TURN_LEFT, state=state)


def test_accelerate_from_rest_outputs_nonzero_u_ref() -> None:
    skills = build_default_skill_library(max_duration=10)
    mgr = SkillRuntimeManager(
        skills,
        default_ctx={
            "action_limit": 1.0,
            "target_speed": 1.0,
            "accelerate_delta_speed": 0.3,
            "accelerate_speed_kp": 1.0,
        },
    )
    state = _make_state(position=[0.0, 0.0], velocity=[0.0, 0.0], goal=[10.0, 0.0], agent_id=0)
    obs = _make_obs(state)
    mgr.activate_skill(agent_id=0, skill_id=SKILL_ACCELERATE, state=state)
    out = mgr.step(agent_id=0, state=state, obs_low=obs, executed_action=np.zeros(2, dtype=np.float32))
    assert float(np.linalg.norm(out.u_ref_skill)) > 1e-4


def test_accelerate_defaults_to_goal_heading() -> None:
    skills = build_default_skill_library(max_duration=10)
    mgr = SkillRuntimeManager(
        skills,
        default_ctx={
            "action_limit": 2.0,
            "target_speed": 1.2,
            "accelerate_delta_speed": 0.4,
            "accelerate_speed_kp": 1.0,
            "accelerate_heading_blend": 0.0,
        },
    )
    state = _make_state(position=[0.0, 0.0], velocity=[1.0, 0.0], goal=[0.0, 10.0], agent_id=0)
    obs = _make_obs(state)
    mgr.activate_skill(agent_id=0, skill_id=SKILL_ACCELERATE, state=state)
    out = mgr.step(agent_id=0, state=state, obs_low=obs, executed_action=np.zeros(2, dtype=np.float32))
    assert out.u_ref_skill[1] > 0.0


def test_cruise_holds_activation_speed() -> None:
    skills = build_default_skill_library(max_duration=10)
    mgr = SkillRuntimeManager(
        skills,
        default_ctx={
            "goal_threshold": 0.1,
            "action_limit": 1.0,
            "cruise_speed_kp": 1.0,
        },
    )
    state = _make_state(position=[0.0, 0.0], velocity=[0.4, 0.0], goal=[10.0, 0.0], agent_id=0)
    obs = _make_obs(state)
    mgr.activate_skill(agent_id=0, skill_id=SKILL_CRUISE, state=state)
    out = mgr.step(agent_id=0, state=state, obs_low=obs, executed_action=np.zeros(2, dtype=np.float32))

    assert out.u_ref_skill.shape == (2,)
    assert abs(float(out.u_ref_skill[0])) < 1e-3


def test_decelerate_zero_velocity_deadzone() -> None:
    skills = build_default_skill_library(max_duration=10)
    mgr = SkillRuntimeManager(
        skills,
        default_ctx={
            "decelerate_init_min_speed": 0.0,
            "decelerate_stop_eps": 0.05,
            "decelerate_kv": 1.5,
            "action_limit": 2.0,
        },
    )
    state = _make_state(position=[0.0, 0.0], velocity=[0.01, 0.0], goal=[1.0, 0.0], agent_id=0)
    obs = _make_obs(state)
    mgr.activate_skill(agent_id=0, skill_id=SKILL_DECELERATE, state=state)
    out = mgr.step(agent_id=0, state=state, obs_low=obs, executed_action=np.zeros(2, dtype=np.float32))
    assert float(np.linalg.norm(out.u_ref_skill)) <= 1e-6
