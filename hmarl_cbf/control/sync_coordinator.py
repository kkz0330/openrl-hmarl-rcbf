from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class SyncStepResult:
    t: int
    k: int
    sync_switch: bool
    tau: Dict[int, int]
    switch_agents: list[int]
    option_k: Dict[int, int]


class SyncCoordinator:
    """
    Option-switch coordinator.

    - mode='sync':
        sync_switch = all_i(beta_i == 1) or max_i(tau_i) >= T_sync_max
        and all agents switch together.
    - mode='async':
        each agent i switches when beta_i == 1 or tau_i >= T_sync_max.
    """

    def __init__(self, num_agents: int, t_sync_max: int, mode: str = "sync") -> None:
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if t_sync_max <= 0:
            raise ValueError("t_sync_max must be positive")
        mode = str(mode).strip().lower()
        if mode not in {"sync", "async"}:
            raise ValueError("mode must be one of {'sync', 'async'}")
        self.num_agents = num_agents
        self.t_sync_max = t_sync_max
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.k = 0
        self.t_k = 0
        self.tau = {i: 0 for i in range(self.num_agents)}
        self.option_k = {i: 0 for i in range(self.num_agents)}

    def step(self, beta_flags: Dict[int, bool]) -> SyncStepResult:
        for agent_id in range(self.num_agents):
            if agent_id not in beta_flags:
                raise KeyError(f"missing beta flag for agent {agent_id}")

        for agent_id in self.tau:
            self.tau[agent_id] += 1

        switch_agents: list[int]
        if self.mode == "sync":
            all_terminated = all(beta_flags.values())
            timeout = max(self.tau.values()) >= self.t_sync_max
            sync_switch = bool(all_terminated or timeout)
            switch_agents = list(self.tau.keys()) if sync_switch else []
        else:
            switch_agents = [
                agent_id
                for agent_id in self.tau
                if bool(beta_flags[agent_id]) or self.tau[agent_id] >= self.t_sync_max
            ]
            sync_switch = bool(len(switch_agents) > 0)

        current_t = self.t
        current_k = self.k
        self.t += 1

        if sync_switch:
            self.k += 1
            self.t_k = self.t
            for agent_id in switch_agents:
                self.tau[agent_id] = 0
                self.option_k[agent_id] += 1

        return SyncStepResult(
            t=current_t,
            k=current_k,
            sync_switch=sync_switch,
            tau=dict(self.tau),
            switch_agents=switch_agents,
            option_k=dict(self.option_k),
        )
