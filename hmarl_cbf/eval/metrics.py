from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class EvalEpisodeStats:
    episode_index: int
    success: bool
    reach_rate: float
    collision_rate: float
    min_h_agent: float
    min_h_obstacle: float
    avg_traj_length: float
    skill_switches: int
    qp_feasible_rate: float
    steps: int
    episode_return_mean: float
    safe_reach_ratio: float = 0.0
    extras: dict = field(default_factory=dict)


@dataclass(slots=True)
class EvalSummary:
    n_episodes: int
    success_rate: float
    reach_rate: float
    collision_rate: float
    safe_reach_ratio: float
    min_h_agent: float
    min_h_obstacle: float
    avg_traj_length: float
    avg_skill_switches: float
    qp_feasible_rate: float
    avg_steps: float
    avg_episode_return: float


def evaluate_summary(episodes: List[EvalEpisodeStats]) -> EvalSummary:
    if len(episodes) == 0:
        return EvalSummary(
            n_episodes=0,
            success_rate=0.0,
            reach_rate=0.0,
            collision_rate=0.0,
            safe_reach_ratio=0.0,
            min_h_agent=0.0,
            min_h_obstacle=0.0,
            avg_traj_length=0.0,
            avg_skill_switches=0.0,
            qp_feasible_rate=0.0,
            avg_steps=0.0,
            avg_episode_return=0.0,
        )

    n = float(len(episodes))
    success_rate = sum(1.0 for e in episodes if e.success) / n
    reach_rate = sum(float(e.reach_rate) for e in episodes) / n
    collision_rate = sum(float(e.collision_rate) for e in episodes) / n
    safe_reach_ratio = sum(float(e.safe_reach_ratio) for e in episodes) / n
    min_h_agent = min(float(e.min_h_agent) for e in episodes)
    min_h_obstacle = min(float(e.min_h_obstacle) for e in episodes)
    avg_traj_length = sum(float(e.avg_traj_length) for e in episodes) / n
    avg_skill_switches = sum(float(e.skill_switches) for e in episodes) / n
    qp_feasible_rate = sum(float(e.qp_feasible_rate) for e in episodes) / n
    avg_steps = sum(float(e.steps) for e in episodes) / n
    avg_episode_return = sum(float(e.episode_return_mean) for e in episodes) / n

    return EvalSummary(
        n_episodes=int(n),
        success_rate=float(success_rate),
        reach_rate=float(reach_rate),
        collision_rate=float(collision_rate),
        safe_reach_ratio=float(safe_reach_ratio),
        min_h_agent=float(min_h_agent),
        min_h_obstacle=float(min_h_obstacle),
        avg_traj_length=float(avg_traj_length),
        avg_skill_switches=float(avg_skill_switches),
        qp_feasible_rate=float(qp_feasible_rate),
        avg_steps=float(avg_steps),
        avg_episode_return=float(avg_episode_return),
    )
