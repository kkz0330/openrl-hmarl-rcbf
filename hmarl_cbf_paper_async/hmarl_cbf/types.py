from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np


TensorLike = Any


def _as_vec2(value: np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {arr.shape}")
    return arr


@dataclass(slots=True)
class AgentState:
    agent_id: int
    position: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray
    radius: float = 0.2

    def __post_init__(self) -> None:
        self.position = _as_vec2(self.position, "position")
        self.velocity = _as_vec2(self.velocity, "velocity")
        self.goal = _as_vec2(self.goal, "goal")
        if self.radius <= 0:
            raise ValueError("radius must be positive")


@dataclass(slots=True)
class LidarScan:
    ranges: np.ndarray
    max_range: float
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        self.ranges = np.asarray(self.ranges, dtype=np.float32).reshape(-1)
        if self.max_range <= 0:
            raise ValueError("max_range must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0")

    @property
    def normalized(self) -> np.ndarray:
        return np.clip(self.ranges / self.max_range, 0.0, 1.0)


@dataclass(slots=True)
class AgentObsHigh:
    self_state: np.ndarray
    goal_relative: np.ndarray
    neighbor_summary: np.ndarray

    def __post_init__(self) -> None:
        self.self_state = np.asarray(self.self_state, dtype=np.float32).reshape(-1)
        self.goal_relative = _as_vec2(self.goal_relative, "goal_relative")
        self.neighbor_summary = np.asarray(self.neighbor_summary, dtype=np.float32).reshape(-1)


@dataclass(slots=True)
class AgentObsLow:
    self_state: np.ndarray
    goal_relative: np.ndarray
    lidar_scan: LidarScan
    neighbor_summary: np.ndarray

    def __post_init__(self) -> None:
        self.self_state = np.asarray(self.self_state, dtype=np.float32).reshape(-1)
        self.goal_relative = _as_vec2(self.goal_relative, "goal_relative")
        self.neighbor_summary = np.asarray(self.neighbor_summary, dtype=np.float32).reshape(-1)

    @property
    def flat(self) -> np.ndarray:
        return np.concatenate(
            [
                self.self_state,
                self.goal_relative,
                self.lidar_scan.normalized,
                self.neighbor_summary,
            ],
            axis=0,
        )


@dataclass(slots=True)
class SkillSpec:
    """Skill 7-tuple schema."""

    skill_id: int
    name: str
    initiation_set_fn: Callable[[AgentState, Dict[str, Any]], bool]
    termination_set_fn: Callable[[AgentState, Dict[str, Any]], bool]
    max_duration: int
    termination_fn: Callable[[AgentState, Dict[str, Any], int], bool]
    safety_constraints_fn: Callable[[AgentState, Dict[str, Any]], Dict[str, Any]]
    intrinsic_reward_fn: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], float]
    safe_skill_policy: Callable[[AgentObsLow, Dict[str, Any]], np.ndarray]

    def __post_init__(self) -> None:
        if self.skill_id < 0:
            raise ValueError("skill_id must be >= 0")
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.max_duration <= 0:
            raise ValueError("max_duration must be positive")
        for fn_name in [
            "initiation_set_fn",
            "termination_set_fn",
            "termination_fn",
            "safety_constraints_fn",
            "intrinsic_reward_fn",
            "safe_skill_policy",
        ]:
            if not callable(getattr(self, fn_name)):
                raise ValueError(f"{fn_name} must be callable")


@dataclass(slots=True)
class QPParam:
    H_mat: TensorLike
    f_lin: TensorLike
    w_clf: TensorLike
    w_cbf: TensorLike
    cbf_slack_max: TensorLike
    cbf_k0: TensorLike
    cbf_k1: TensorLike
    clf_k: TensorLike
    hocbf_gamma_h: TensorLike | None = None
    hocbf_gamma_hdot: TensorLike | None = None


@dataclass(slots=True)
class QPProblem:
    H_mat: TensorLike
    f_lin: TensorLike
    w_clf: TensorLike
    w_cbf: TensorLike
    cbf_slack_max: TensorLike
    A_cbf: TensorLike
    b_cbf: TensorLike
    A_clf: TensorLike
    b_clf: TensorLike
    u_min: TensorLike
    u_max: TensorLike
    delta_min: float = 0.0


@dataclass(slots=True)
class QPSolution:
    action: TensorLike
    slack: TensorLike
    objective: TensorLike
    feasible: bool
    solver_status: str
    cbf_slack: TensorLike | None = None


@dataclass(slots=True)
class SafetyMetrics:
    min_h_agent: float
    min_h_obstacle: float
    collision: bool
    reached_goal: bool
    qp_feasible: bool


@dataclass(slots=True)
class LowStepTransition:
    t: int
    agent_id: int
    obs_low: AgentObsLow
    skill_id: int
    option_k: int
    action: np.ndarray
    reward_int: float
    reward_ext: float
    done: bool
    logp: float | None = None
    value: float | None = None
    advantage: float | None = None
    return_target: float | None = None
    sync_switch: bool = False
    terminated_by_skill: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action = _as_vec2(self.action, "action")


@dataclass(slots=True)
class HighOptionTransition:
    k: int
    agent_id: int
    t_start: int
    t_end: int
    obs_high: AgentObsHigh
    skill_id: int
    logp: float
    value: float
    return_ext: float
    done: bool
    advantage: float | None = None
    value_target: float | None = None
    sync_switch: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.t_end < self.t_start:
            raise ValueError("t_end must be >= t_start")
