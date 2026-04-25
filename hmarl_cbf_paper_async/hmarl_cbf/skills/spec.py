from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from hmarl_cbf.skills.library import build_default_skill_library
from hmarl_cbf.types import LIDAR_HIT_NONE, AgentObsLow, AgentState, LidarScan, SkillSpec


def validate_skill_spec(skill: SkillSpec) -> None:
    """Runs minimal schema-level validation for a skill specification."""
    probe_state = AgentState(
        agent_id=0,
        position=np.zeros(2, dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        goal=np.ones(2, dtype=np.float32),
    )
    probe_obs = AgentObsLow(
        self_state=np.zeros(4, dtype=np.float32),
        goal_relative=np.ones(2, dtype=np.float32),
        lidar_scan=LidarScan(
            ranges=np.ones(8, dtype=np.float32),
            max_range=8.0,
            angles=np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32),
            origin=np.zeros(2, dtype=np.float32),
            hit_points=np.zeros((8, 2), dtype=np.float32),
            hit_valid=np.zeros(8, dtype=np.bool_),
            hit_kinds=np.full(8, LIDAR_HIT_NONE, dtype=np.int32),
        ),
        neighbor_summary=np.zeros(8, dtype=np.float32),
    )
    ctx: Dict[str, Any] = {"dt": 0.03}
    _ = bool(skill.initiation_set_fn(probe_state, ctx))
    _ = bool(skill.termination_set_fn(probe_state, ctx))
    _ = bool(skill.termination_fn(probe_state, ctx, 1))
    _ = skill.safety_constraints_fn(probe_state, ctx)
    _ = float(skill.intrinsic_reward_fn(np.zeros(4, dtype=np.float32), np.zeros(2, dtype=np.float32), ctx))
    _ = np.asarray(skill.safe_skill_policy(probe_obs, ctx), dtype=np.float32)

def build_validated_skill_library(max_duration: int = 20) -> List[SkillSpec]:
    skills = build_default_skill_library(max_duration=max_duration)
    for skill in skills:
        validate_skill_spec(skill)
    return skills
