from __future__ import annotations

from typing import Any, Dict

from hmarl_cbf.types import AgentState, SkillSpec


def evaluate_termination(skill: SkillSpec, state: AgentState, ctx: Dict[str, Any], tau: int) -> bool:
    """Unified termination gate used by the runtime manager."""
    local_ctx = dict(ctx)
    local_ctx["max_duration"] = int(skill.max_duration)
    return bool(skill.termination_fn(state, local_ctx, tau))
