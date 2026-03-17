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
)
from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.policies import HighLevelPolicy, LowLevelQPPolicy
from hmarl_cbf.skills import SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.train import TrainerHooks, TrainerSyncOnPolicy


def test_trainer_evaluate_returns_metrics() -> None:
    env = MultiUAV2DEnv(
        n_agents=2,
        n_obstacles=1,
        horizon=30,
        lidar_beams=8,
        max_neighbors=2,
        terminate_on_collision=False,
    )
    high_policy = HighLevelPolicy(obs_dim=14, n_skills=6, hidden_dim=32)
    low_policy = LowLevelQPPolicy(obs_dim=22, n_skills=6, action_dim=2, hidden_dim=32)
    runtime = SkillRuntimeManager(
        build_default_skill_library(max_duration=8),
        default_ctx={"turn_min_speed": 0.0, "goal_threshold": 0.3},
    )
    constraint_builder = ConstraintBuilder(d_min_agent=0.6, d_safe_obs=0.6)
    low_controller = LowLevelSafeController(
        low_policy=low_policy,
        constraint_builder=constraint_builder,
        qp_solver=DifferentiableQPSolver(use_stub_if_unavailable=True),
        skill_ref_weight=0.7,
    )
    trainer = TrainerSyncOnPolicy(
        env=env,
        high_policy=high_policy,
        low_policy=low_policy,
        constraint_builder=constraint_builder,
        qp_solver=DifferentiableQPSolver(use_stub_if_unavailable=True),
        coordinator=SyncCoordinator(num_agents=2, t_sync_max=8),
        buffer=HierRolloutBuffer(),
        skill_runtime=runtime,
        low_level_controller=low_controller,
        hooks=TrainerHooks(eval_episodes=1, eval_deterministic=True, eval_render=False),
    )
    metrics = trainer.evaluate()
    assert metrics["eval_n_episodes"] == 1.0
    assert 0.0 <= metrics["eval_success_rate"] <= 1.0
    assert 0.0 <= metrics["eval_safe_reach_ratio"] <= 1.0
    assert 0.0 <= metrics["eval_qp_feasible_rate"] <= 1.0
