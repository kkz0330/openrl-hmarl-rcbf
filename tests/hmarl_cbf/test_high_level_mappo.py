import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hmarl_cbf.high_level import MAPPOConfig, OnPolicyMAPPO
from hmarl_cbf.policies import HighLevelPolicy
from hmarl_cbf.types import AgentObsHigh, HighOptionTransition


def _obs(seed: int) -> AgentObsHigh:
    rng = np.random.default_rng(seed)
    return AgentObsHigh(
        self_state=rng.normal(size=4).astype(np.float32),
        goal_relative=rng.normal(size=2).astype(np.float32),
        neighbor_summary=rng.normal(size=8).astype(np.float32),
    )


def _make_transitions(policy: HighLevelPolicy, n: int = 32) -> list[HighOptionTransition]:
    transitions: list[HighOptionTransition] = []
    for i in range(n):
        obs = _obs(i)
        obs_vec = np.concatenate([obs.self_state, obs.goal_relative, obs.neighbor_summary], axis=0)
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = policy.act(obs_t, deterministic=False)
        transitions.append(
            HighOptionTransition(
                k=i // 4,
                agent_id=i % 4,
                t_start=i,
                t_end=i + 2,
                obs_high=obs,
                skill_id=int(out["z"].item()),
                logp=float(out["logp"].item()),
                value=float(out["value"].item()),
                return_ext=float(np.random.normal(loc=0.5, scale=0.2)),
                done=False,
            )
        )
    return transitions


def test_mappo_update_changes_policy_params() -> None:
    policy = HighLevelPolicy(obs_dim=14, n_skills=6, hidden_dim=64)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    updater = OnPolicyMAPPO(
        policy=policy,
        optimizer=opt,
        config=MAPPOConfig(ppo_epochs=2, minibatch_size=16),
    )
    transitions = _make_transitions(policy, n=40)

    before = policy.actor_head.weight.detach().clone()
    metrics = updater.update(transitions)
    after = policy.actor_head.weight.detach().clone()

    assert metrics["n_samples"] == 40.0
    assert np.isfinite(metrics["loss_total"])
    assert not torch.equal(before, after)


def test_mappo_empty_transitions_returns_zero_metrics() -> None:
    policy = HighLevelPolicy(obs_dim=14, n_skills=6, hidden_dim=32)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    updater = OnPolicyMAPPO(policy=policy, optimizer=opt)
    metrics = updater.update([])
    assert metrics["n_samples"] == 0.0
    assert metrics["loss_total"] == 0.0
