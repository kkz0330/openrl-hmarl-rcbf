from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from hmarl_cbf.env.lidar import LidarModel
from hmarl_cbf.types import AgentObsHigh, AgentObsLow, AgentState


class ObservationBuilder:
    """Builds partially observable high/low-level observations."""

    def __init__(self, lidar: LidarModel, neighbor_radius: float = 4.0, max_neighbors: int = 4) -> None:
        self.lidar = lidar
        self.neighbor_radius = float(neighbor_radius)
        self.max_neighbors = int(max_neighbors)

    def _neighbor_summary(self, states: List[AgentState], i: int) -> Tuple[np.ndarray, List[AgentState]]:
        state_i = states[i]
        candidates: List[Tuple[float, AgentState]] = []
        for j, state_j in enumerate(states):
            if i == j:
                continue
            delta = state_j.position - state_i.position
            dist = float(np.linalg.norm(delta))
            if dist <= self.neighbor_radius:
                candidates.append((dist, state_j))
        candidates.sort(key=lambda x: x[0])
        selected = [s for _, s in candidates[: self.max_neighbors]]

        summary = np.zeros(self.max_neighbors * 4, dtype=np.float32)
        for idx, neighbor in enumerate(selected):
            base = idx * 4
            rel_pos = neighbor.position - state_i.position
            rel_vel = neighbor.velocity - state_i.velocity
            summary[base : base + 2] = rel_pos
            summary[base + 2 : base + 4] = rel_vel
        return summary, selected

    def build(
        self,
        states: List[AgentState],
        obstacles: List[Dict[str, np.ndarray | float]],
    ) -> Dict[int, Dict[str, AgentObsHigh | AgentObsLow]]:
        obs: Dict[int, Dict[str, AgentObsHigh | AgentObsLow]] = {}
        for i, state_i in enumerate(states):
            summary, selected = self._neighbor_summary(states, i)
            goal_rel = (state_i.goal - state_i.position).astype(np.float32)
            self_state = np.concatenate([state_i.position, state_i.velocity], axis=0).astype(np.float32)

            neighbor_circles = [
                {"center": neighbor.position.astype(np.float32), "radius": float(neighbor.radius)} for neighbor in selected
            ]
            lidar_scan = self.lidar.scan(origin=state_i.position, obstacles=obstacles, neighbors=neighbor_circles)

            obs_high = AgentObsHigh(
                self_state=self_state,
                goal_relative=goal_rel,
                neighbor_summary=summary,
            )
            obs_low = AgentObsLow(
                self_state=self_state,
                goal_relative=goal_rel,
                lidar_scan=lidar_scan,
                neighbor_summary=summary,
            )
            obs[state_i.agent_id] = {"high": obs_high, "low": obs_low}
        return obs
