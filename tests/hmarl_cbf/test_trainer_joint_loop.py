import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cvxpy")
pytest.importorskip("cvxpylayers")

from hmarl_cbf.buffer import HierRolloutBuffer
from hmarl_cbf.control import (
    ConstraintBuilder,
    DifferentiableQPSolver,
    LowLevelSafeController,
    SyncCoordinator,
    TorchDifferentiableQPSolver,
)
from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.high_level import MAPPOConfig, OnPolicyMAPPO
from hmarl_cbf.policies import HighLevelPolicy, LowLevelQPPolicy
from hmarl_cbf.skills import SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.train import TrainerHooks, TrainerSyncOnPolicy


def test_trainer_joint_sync_loop_one_iteration() -> None:
    env = MultiUAV2DEnv(
        n_agents=2,
        n_obstacles=1,
        horizon=40,
        lidar_beams=8,
        max_neighbors=2,
        terminate_on_collision=False,
    )
    high_policy = HighLevelPolicy(obs_dim=14, n_skills=6, hidden_dim=64)
    low_policy = LowLevelQPPolicy(obs_dim=22, n_skills=6, action_dim=2, hidden_dim=64)

    runtime = SkillRuntimeManager(
        build_default_skill_library(max_duration=8),
        default_ctx={"turn_min_speed": 0.0, "goal_threshold": 0.3},
    )
    constraint_builder = ConstraintBuilder(d_min_agent=0.6, d_safe_obs=0.6, u_min=[-1.0, -1.0], u_max=[1.0, 1.0])
    qp_solver = DifferentiableQPSolver(use_stub_if_unavailable=True)
    low_controller = LowLevelSafeController(
        low_policy=low_policy,
        constraint_builder=constraint_builder,
        qp_solver=qp_solver,
        skill_ref_weight=0.7,
    )
    high_updater = OnPolicyMAPPO(
        policy=high_policy,
        optimizer=torch.optim.Adam(high_policy.parameters(), lr=3e-4),
        config=MAPPOConfig(ppo_epochs=1, minibatch_size=8),
    )
    low_opt = torch.optim.Adam(low_policy.parameters(), lr=1e-3)
    trainer = TrainerSyncOnPolicy(
        env=env,
        high_policy=high_policy,
        low_policy=low_policy,
        constraint_builder=constraint_builder,
        qp_solver=qp_solver,
        coordinator=SyncCoordinator(num_agents=2, t_sync_max=8),
        buffer=HierRolloutBuffer(),
        skill_runtime=runtime,
        low_level_controller=low_controller,
        diff_qp_solver=TorchDifferentiableQPSolver(action_dim=2),
        high_level_updater=high_updater,
        low_level_optimizer=low_opt,
        hooks=TrainerHooks(
            rollout_steps=10,
            low_update_epochs=1,
            low_max_samples_per_iter=32,
            low_target_step_scale=0.02,
        ),
    )

    rollout_stats = trainer.collect_rollout()
    assert rollout_stats["steps_collected"] > 0
    assert rollout_stats["high_samples"] > 0
    assert rollout_stats["low_samples"] > 0

    low_stats = trainer.update_low_level()
    assert low_stats["n_updates"] > 0

    high_stats = trainer.update_high_level()
    assert high_stats["n_samples"] > 0
