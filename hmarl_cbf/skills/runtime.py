from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

import numpy as np

from hmarl_cbf.skills.library import (
    SKILL_TURN_LEFT,
    SKILL_TURN_RIGHT,
)
from hmarl_cbf.skills.termination import evaluate_termination
from hmarl_cbf.types import AgentObsLow, AgentState, SkillSpec


@dataclass(slots=True)
class SkillStepOutput:
    agent_id: int
    skill_id: int
    tau: int
    beta: bool
    intrinsic_reward: float
    u_ref_skill: np.ndarray
    safety_constraints: Dict[str, Any]


class SkillRuntimeManager:
    """
    Per-agent skill runtime for synchronous execution.

    Responsibilities:
    - activate skill options per agent
    - maintain per-agent in-skill timer tau
    - evaluate beta termination
    - emit per-step intrinsic reward and safe skill reference action
    """

    def __init__(
        self,
        skills: Iterable[SkillSpec],
        default_ctx: Mapping[str, Any] | None = None,
    ) -> None:
        self.skill_by_id: Dict[int, SkillSpec] = {skill.skill_id: skill for skill in skills}
        if not self.skill_by_id:
            raise ValueError("at least one skill is required")
        self.default_ctx = dict(default_ctx or {})
        self._active_skill: Dict[int, int] = {}
        self._tau: Dict[int, int] = {}
        self._agent_ctx: Dict[int, Dict[str, Any]] = {}

    def reset(self, agent_ids: Iterable[int]) -> None:
        for agent_id in agent_ids:
            self._active_skill.pop(agent_id, None)
            self._tau.pop(agent_id, None)
            self._agent_ctx.pop(agent_id, None)

    def current_skill_id(self, agent_id: int) -> int:
        if agent_id not in self._active_skill:
            raise KeyError(f"agent {agent_id} has no active skill")
        return self._active_skill[agent_id]

    def current_tau(self, agent_id: int) -> int:
        return int(self._tau.get(agent_id, 0))

    def _heading_from_state(self, state: AgentState) -> float:
        vel_norm = float(np.linalg.norm(state.velocity))
        if vel_norm > 1e-6:
            return float(atan2(float(state.velocity[1]), float(state.velocity[0])))
        goal_vec = state.goal - state.position
        return float(atan2(float(goal_vec[1]), float(goal_vec[0])))

    def _prepare_agent_ctx(
        self,
        agent_id: int,
        skill: SkillSpec,
        state: AgentState,
        extra_ctx: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        ctx: Dict[str, Any] = dict(self.default_ctx)
        if extra_ctx:
            ctx.update(dict(extra_ctx))
        ctx["max_duration"] = int(skill.max_duration)

        heading = self._heading_from_state(state)
        ctx["start_heading"] = heading
        if skill.skill_id == SKILL_TURN_LEFT:
            turn_angle = float(ctx.get("turn_target_angle", 0.6))
            ctx["target_heading"] = heading + turn_angle
            ctx["heading_ref"] = heading + turn_angle
        elif skill.skill_id == SKILL_TURN_RIGHT:
            turn_angle = float(ctx.get("turn_target_angle", 0.6))
            ctx["target_heading"] = heading - turn_angle
            ctx["heading_ref"] = heading - turn_angle
        else:
            ctx["heading_ref"] = heading
        return ctx

    def activate_skill(
        self,
        agent_id: int,
        skill_id: int,
        state: AgentState,
        extra_ctx: Mapping[str, Any] | None = None,
    ) -> None:
        if skill_id not in self.skill_by_id:
            raise KeyError(f"unknown skill_id={skill_id}")
        skill = self.skill_by_id[skill_id]
        ctx = self._prepare_agent_ctx(agent_id, skill, state, extra_ctx=extra_ctx)
        if not skill.initiation_set_fn(state, ctx):
            raise ValueError(f"agent {agent_id} cannot initiate skill {skill.name} under current state")
        self._active_skill[agent_id] = skill_id
        self._tau[agent_id] = 0
        self._agent_ctx[agent_id] = ctx

    def activate_skills(
        self,
        skill_map: Mapping[int, int],
        states: Mapping[int, AgentState],
        extra_ctx: Mapping[str, Any] | None = None,
    ) -> None:
        for agent_id, skill_id in skill_map.items():
            self.activate_skill(agent_id=agent_id, skill_id=skill_id, state=states[agent_id], extra_ctx=extra_ctx)

    def _merge_ctx(self, agent_id: int, runtime_ctx: Mapping[str, Any] | None) -> MutableMapping[str, Any]:
        if agent_id not in self._agent_ctx:
            raise KeyError(f"agent {agent_id} has no runtime skill context")
        ctx: MutableMapping[str, Any] = dict(self._agent_ctx[agent_id])
        if runtime_ctx:
            ctx.update(dict(runtime_ctx))
        return ctx

    @staticmethod
    def _state_to_reward_vec(state: AgentState) -> np.ndarray:
        goal_rel = state.goal - state.position
        return np.concatenate([state.position, state.velocity, goal_rel], axis=0).astype(np.float32)

    def control_target(
        self,
        agent_id: int,
        state: AgentState,
        obs_low: AgentObsLow,
        runtime_ctx: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if agent_id not in self._active_skill:
            raise KeyError(f"agent {agent_id} has no active skill; call activate_skill first")
        skill_id = self._active_skill[agent_id]
        skill = self.skill_by_id[skill_id]
        ctx = self._merge_ctx(agent_id, runtime_ctx)
        ctx["goal_relative"] = (state.goal - state.position).astype(np.float32)
        return {
            "skill_id": skill_id,
            "tau": int(self._tau.get(agent_id, 0)),
            "u_ref_skill": np.asarray(skill.safe_skill_policy(obs_low, dict(ctx)), dtype=np.float32).reshape(2),
            "safety_constraints": dict(skill.safety_constraints_fn(state, dict(ctx))),
        }

    def control_targets(
        self,
        states: Mapping[int, AgentState],
        obs_low: Mapping[int, AgentObsLow],
        runtime_ctx: Mapping[str, Any] | None = None,
    ) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for agent_id, state in states.items():
            out[agent_id] = self.control_target(
                agent_id=agent_id,
                state=state,
                obs_low=obs_low[agent_id],
                runtime_ctx=runtime_ctx,
            )
        return out

    def step(
        self,
        agent_id: int,
        state: AgentState,
        obs_low: AgentObsLow,
        executed_action: np.ndarray,
        runtime_ctx: Mapping[str, Any] | None = None,
    ) -> SkillStepOutput:
        if agent_id not in self._active_skill:
            raise KeyError(f"agent {agent_id} has no active skill; call activate_skill first")

        skill_id = self._active_skill[agent_id]
        skill = self.skill_by_id[skill_id]
        tau_next = int(self._tau.get(agent_id, 0)) + 1
        ctx = self._merge_ctx(agent_id, runtime_ctx)
        ctx["goal_relative"] = (state.goal - state.position).astype(np.float32)

        beta = evaluate_termination(skill=skill, state=state, ctx=dict(ctx), tau=tau_next)
        reward_vec = self._state_to_reward_vec(state)
        intrinsic_reward = float(skill.intrinsic_reward_fn(reward_vec, np.asarray(executed_action, dtype=np.float32), dict(ctx)))
        u_ref_skill = np.asarray(skill.safe_skill_policy(obs_low, dict(ctx)), dtype=np.float32).reshape(2)
        safety_constraints = dict(skill.safety_constraints_fn(state, dict(ctx)))

        self._tau[agent_id] = tau_next
        if beta:
            # Keep the context for post-step inspection; caller decides when to activate next option.
            pass

        return SkillStepOutput(
            agent_id=agent_id,
            skill_id=skill_id,
            tau=tau_next,
            beta=beta,
            intrinsic_reward=intrinsic_reward,
            u_ref_skill=u_ref_skill,
            safety_constraints=safety_constraints,
        )

    def step_all(
        self,
        states: Mapping[int, AgentState],
        obs_low: Mapping[int, AgentObsLow],
        executed_actions: Mapping[int, np.ndarray],
        runtime_ctx: Mapping[str, Any] | None = None,
    ) -> Dict[int, SkillStepOutput]:
        outputs: Dict[int, SkillStepOutput] = {}
        for agent_id in states:
            outputs[agent_id] = self.step(
                agent_id=agent_id,
                state=states[agent_id],
                obs_low=obs_low[agent_id],
                executed_action=np.asarray(executed_actions[agent_id], dtype=np.float32),
                runtime_ctx=runtime_ctx,
            )
        return outputs
