from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class SyncStepResult:
    t: int
    k: int
    sync_switch: bool
    tau: Dict[int, int]


class SyncCoordinator:
    """
    Synchronous option-switch coordinator.

    Rule:
        sync_switch = all_i(beta_i == 1) or max_i(tau_i) >= T_sync_max
    """

    def __init__(self, num_agents: int, t_sync_max: int) -> None:
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if t_sync_max <= 0:
            raise ValueError("t_sync_max must be positive")
        self.num_agents = num_agents
        self.t_sync_max = t_sync_max
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.k = 0
        self.t_k = 0
        self.tau = {i: 0 for i in range(self.num_agents)}

    def step(self, beta_flags: Dict[int, bool]) -> SyncStepResult:
        for agent_id in range(self.num_agents):
            if agent_id not in beta_flags:
                raise KeyError(f"missing beta flag for agent {agent_id}")

        for agent_id in self.tau:
            self.tau[agent_id] += 1

        all_terminated = all(beta_flags.values())
        timeout = max(self.tau.values()) >= self.t_sync_max
        sync_switch = bool(all_terminated or timeout)

        current_t = self.t
        current_k = self.k
        self.t += 1

        if sync_switch:
            self.k += 1
            self.t_k = self.t
            for agent_id in self.tau:
                self.tau[agent_id] = 0

        return SyncStepResult(
            t=current_t,
            k=current_k,
            sync_switch=sync_switch,
            tau=dict(self.tau),
        )
