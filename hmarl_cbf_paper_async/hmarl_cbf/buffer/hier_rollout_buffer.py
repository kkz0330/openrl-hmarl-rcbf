from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from hmarl_cbf.types import HighOptionTransition, LowStepTransition


@dataclass(slots=True)
class _OpenHighOption:
    k: int
    agent_id: int
    t_start: int
    obs_high: Any
    skill_id: int
    logp: float
    value: float
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HierRolloutBuffer:
    low_steps: List[LowStepTransition] = field(default_factory=list)
    high_options: List[HighOptionTransition] = field(default_factory=list)
    _open_high: Dict[int, _OpenHighOption] = field(default_factory=dict)

    def add_low_step(self, transition: LowStepTransition) -> None:
        self.low_steps.append(transition)

    def add_high_option(self, transition: HighOptionTransition) -> None:
        self.high_options.append(transition)

    def start_high_option(
        self,
        k: int,
        agent_id: int,
        t_start: int,
        obs_high: Any,
        skill_id: int,
        logp: float,
        value: float,
        info: Dict[str, Any] | None = None,
    ) -> None:
        if agent_id in self._open_high:
            raise RuntimeError(f"agent {agent_id} already has an open high option")
        self._open_high[agent_id] = _OpenHighOption(
            k=k,
            agent_id=agent_id,
            t_start=t_start,
            obs_high=obs_high,
            skill_id=skill_id,
            logp=float(logp),
            value=float(value),
            info=dict(info or {}),
        )

    def close_high_option(
        self,
        agent_id: int,
        t_end: int,
        return_ext: float,
        done: bool,
        sync_switch: bool,
        info: Dict[str, Any] | None = None,
    ) -> HighOptionTransition:
        if agent_id not in self._open_high:
            raise RuntimeError(f"agent {agent_id} has no open high option")
        opened = self._open_high.pop(agent_id)
        merged_info = dict(opened.info)
        merged_info.update(dict(info or {}))
        transition = HighOptionTransition(
            k=opened.k,
            agent_id=agent_id,
            t_start=opened.t_start,
            t_end=t_end,
            obs_high=opened.obs_high,
            skill_id=opened.skill_id,
            logp=opened.logp,
            value=opened.value,
            return_ext=float(return_ext),
            done=bool(done),
            sync_switch=bool(sync_switch),
            info=merged_info,
        )
        self.high_options.append(transition)
        return transition

    def has_open_high_option(self, agent_id: int) -> bool:
        return agent_id in self._open_high

    def clear(self) -> None:
        self.low_steps.clear()
        self.high_options.clear()
        self._open_high.clear()

    def clear_low(self) -> None:
        self.low_steps.clear()

    def clear_high(self) -> None:
        self.high_options.clear()
        self._open_high.clear()

    def size_low(self) -> int:
        return len(self.low_steps)

    def size_high(self) -> int:
        return len(self.high_options)

    def snapshot(self) -> tuple[List[LowStepTransition], List[HighOptionTransition]]:
        return list(self.low_steps), list(self.high_options)

    def high_by_agent(self) -> Dict[int, List[HighOptionTransition]]:
        out: Dict[int, List[HighOptionTransition]] = {}
        for tr in self.high_options:
            out.setdefault(tr.agent_id, []).append(tr)
        for agent_id in out:
            out[agent_id].sort(key=lambda x: (x.k, x.t_start))
        return out

    def high_advantages_by_option(self) -> Dict[Tuple[int, int], float]:
        out: Dict[Tuple[int, int], float] = {}
        for tr in self.high_options:
            out[(int(tr.agent_id), int(tr.k))] = float(tr.advantage if tr.advantage is not None else 0.0)
        return out

    def low_by_agent(self) -> Dict[int, List[LowStepTransition]]:
        out: Dict[int, List[LowStepTransition]] = {}
        for tr in self.low_steps:
            out.setdefault(tr.agent_id, []).append(tr)
        for agent_id in out:
            out[agent_id].sort(key=lambda x: x.t)
        return out

    def compute_high_advantages(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        use_gae: bool = True,
        bootstrap_value_by_agent: Dict[int, float] | None = None,
    ) -> Dict[str, float]:
        bootstrap_value_by_agent = dict(bootstrap_value_by_agent or {})
        by_agent = self.high_by_agent()
        n = 0
        adv_vals: List[float] = []
        ret_vals: List[float] = []

        for agent_id, seq in by_agent.items():
            if len(seq) == 0:
                continue

            if use_gae:
                gae = 0.0
                for idx in reversed(range(len(seq))):
                    tr = seq[idx]
                    if idx + 1 < len(seq):
                        next_value = float(seq[idx + 1].value)
                        nonterminal = 0.0 if tr.done else 1.0
                    else:
                        next_value = float(bootstrap_value_by_agent.get(agent_id, 0.0))
                        nonterminal = 0.0 if tr.done else 1.0
                    delta = float(tr.return_ext) + gamma * next_value * nonterminal - float(tr.value)
                    gae = delta + gamma * lam * nonterminal * gae
                    tr.advantage = float(gae)
                    tr.value_target = float(gae + tr.value)
                    adv_vals.append(float(tr.advantage))
                    ret_vals.append(float(tr.value_target))
                    n += 1
            else:
                running = float(bootstrap_value_by_agent.get(agent_id, 0.0))
                for tr in reversed(seq):
                    if tr.done:
                        running = 0.0
                    running = float(tr.return_ext) + gamma * running
                    tr.value_target = float(running)
                    tr.advantage = float(running - tr.value)
                    adv_vals.append(float(tr.advantage))
                    ret_vals.append(float(tr.value_target))
                    n += 1

        if n == 0:
            return {"n_samples": 0.0, "adv_mean": 0.0, "adv_std": 0.0, "value_target_mean": 0.0}
        adv_arr = adv_vals
        ret_arr = ret_vals
        adv_mean = sum(adv_arr) / len(adv_arr)
        adv_std = (sum((x - adv_mean) ** 2 for x in adv_arr) / len(adv_arr)) ** 0.5
        ret_mean = sum(ret_arr) / len(ret_arr)
        return {
            "n_samples": float(n),
            "adv_mean": float(adv_mean),
            "adv_std": float(adv_std),
            "value_target_mean": float(ret_mean),
        }

    def compute_low_returns(
        self,
        gamma: float = 0.99,
        ext_reward_coef: float = 0.0,
        reset_on_sync_switch: bool = True,
        reward_mix_eta: float = 0.0,
        high_adv_by_option: Dict[Tuple[int, int], float] | None = None,
        divide_high_adv_by_n_agents: bool = True,
        n_agents: int = 1,
        safety_margin_coef: float = 0.0,
        safety_margin_h_agent: float = 0.0,
        safety_margin_h_obstacle: float = 0.0,
    ) -> Dict[str, float]:
        high_adv_by_option = dict(high_adv_by_option or {})
        by_agent = self.low_by_agent()
        n = 0
        ret_vals: List[float] = []
        adv_vals: List[float] = []
        mix_vals: List[float] = []
        high_mix_vals: List[float] = []
        safety_penalty_vals: List[float] = []

        eta = float(max(0.0, min(1.0, reward_mix_eta)))
        denom = float(max(1, n_agents)) if divide_high_adv_by_n_agents else 1.0
        margin_coef = float(max(0.0, safety_margin_coef))
        h_agent_margin = float(max(0.0, safety_margin_h_agent))
        h_obs_margin = float(max(0.0, safety_margin_h_obstacle))

        for _, seq in by_agent.items():
            running = 0.0
            for tr in reversed(seq):
                if tr.done or (reset_on_sync_switch and tr.sync_switch):
                    running = 0.0
                high_adv = float(high_adv_by_option.get((int(tr.agent_id), int(tr.option_k)), 0.0))
                high_term = high_adv / denom
                min_h_agent = float(tr.info.get("min_h_agent", 0.0))
                min_h_obstacle = float(tr.info.get("min_h_obstacle", 0.0))
                safety_penalty = 0.0
                if margin_coef > 0.0:
                    if h_agent_margin > 0.0 and min_h_agent < h_agent_margin:
                        safety_penalty += (h_agent_margin - min_h_agent) ** 2
                    if h_obs_margin > 0.0 and min_h_obstacle < h_obs_margin:
                        safety_penalty += (h_obs_margin - min_h_obstacle) ** 2
                    safety_penalty *= margin_coef
                reward = (
                    eta * high_term
                    + (1.0 - eta) * float(tr.reward_int)
                    + float(ext_reward_coef) * float(tr.reward_ext)
                    - float(safety_penalty)
                )
                running = reward + gamma * running
                tr.return_target = float(running)
                if tr.value is not None:
                    tr.advantage = float(running - float(tr.value))
                else:
                    tr.advantage = float(running)
                ret_vals.append(float(tr.return_target))
                adv_vals.append(float(tr.advantage))
                mix_vals.append(float(reward))
                high_mix_vals.append(float(high_term))
                safety_penalty_vals.append(float(safety_penalty))
                n += 1

        if n == 0:
            return {
                "n_samples": 0.0,
                "return_mean": 0.0,
                "adv_mean": 0.0,
                "reward_mix_mean": 0.0,
                "high_adv_mix_mean": 0.0,
                "safety_margin_penalty_mean": 0.0,
            }
        return {
            "n_samples": float(n),
            "return_mean": float(sum(ret_vals) / len(ret_vals)),
            "adv_mean": float(sum(adv_vals) / len(adv_vals)),
            "reward_mix_mean": float(sum(mix_vals) / len(mix_vals)),
            "high_adv_mix_mean": float(sum(high_mix_vals) / len(high_mix_vals)),
            "safety_margin_penalty_mean": float(sum(safety_penalty_vals) / max(1, len(safety_penalty_vals))),
        }
