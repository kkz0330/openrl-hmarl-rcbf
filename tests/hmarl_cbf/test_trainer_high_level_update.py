import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hmarl_cbf.buffer import HierRolloutBuffer
from hmarl_cbf.control import ConstraintBuilder, DifferentiableQPSolver, SyncCoordinator
from hmarl_cbf.high_level import MAPPOConfig, OnPolicyMAPPO
from hmarl_cbf.policies import HighLevelPolicy
from hmarl_cbf.train import TrainerSyncOnPolicy
from hmarl_cbf.types import AgentObsHigh, HighOptionTransition


def _obs(seed: int) -> AgentObsHigh:
    rng = np.random.default_rng(seed)
    return AgentObsHigh(
        self_state=rng.normal(size=4).astype(np.float32),
        goal_relative=rng.normal(size=2).astype(np.float32),
        neighbor_summary=rng.normal(size=8).astype(np.float32),
    )


def test_trainer_update_high_level_consumes_buffer() -> None:
    policy = HighLevelPolicy(obs_dim=14, n_skills=6, hidden_dim=64)
    updater = OnPolicyMAPPO(
        policy=policy,
        optimizer=torch.optim.Adam(policy.parameters(), lr=3e-4),
        config=MAPPOConfig(ppo_epochs=1, minibatch_size=8),
    )
    buffer = HierRolloutBuffer()

    for i in range(16):
        obs = _obs(i)
        obs_vec = np.concatenate([obs.self_state, obs.goal_relative, obs.neighbor_summary], axis=0)
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act = policy.act(obs_t)
        buffer.add_high_option(
            HighOptionTransition(
                k=i // 4,
                agent_id=i % 4,
                t_start=i,
                t_end=i + 1,
                obs_high=obs,
                skill_id=int(act["z"].item()),
                logp=float(act["logp"].item()),
                value=float(act["value"].item()),
                return_ext=float(np.random.normal(loc=0.4, scale=0.1)),
                done=False,
            )
        )

    trainer = TrainerSyncOnPolicy(
        env=object(),
        high_policy=policy,
        low_policy=object(),
        constraint_builder=ConstraintBuilder(),
        qp_solver=DifferentiableQPSolver(use_stub_if_unavailable=True),
        coordinator=SyncCoordinator(num_agents=2, t_sync_max=5),
        buffer=buffer,
        high_level_updater=updater,
    )

    assert buffer.size_high() == 16
    metrics = trainer.update_high_level()
    assert np.isfinite(metrics["loss_total"])
    assert buffer.size_high() == 0
