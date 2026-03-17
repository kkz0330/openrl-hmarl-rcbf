from hmarl_cbf.eval import EvalEpisodeStats, evaluate_summary


def test_evaluate_summary_aggregates() -> None:
    episodes = [
        EvalEpisodeStats(
            episode_index=0,
            success=True,
            reach_rate=1.0,
            collision_rate=0.0,
            min_h_agent=0.2,
            min_h_obstacle=0.1,
            avg_traj_length=3.0,
            skill_switches=4,
            qp_feasible_rate=1.0,
            steps=20,
            episode_return_mean=1.2,
            safe_reach_ratio=1.0,
        ),
        EvalEpisodeStats(
            episode_index=1,
            success=False,
            reach_rate=0.5,
            collision_rate=0.5,
            min_h_agent=-0.1,
            min_h_obstacle=-0.2,
            avg_traj_length=5.0,
            skill_switches=6,
            qp_feasible_rate=0.8,
            steps=30,
            episode_return_mean=0.4,
            safe_reach_ratio=0.5,
        ),
    ]
    summary = evaluate_summary(episodes)
    assert summary.n_episodes == 2
    assert abs(summary.success_rate - 0.5) < 1e-6
    assert abs(summary.reach_rate - 0.75) < 1e-6
    assert abs(summary.collision_rate - 0.25) < 1e-6
    assert abs(summary.safe_reach_ratio - 0.75) < 1e-6
    assert abs(summary.min_h_agent - (-0.1)) < 1e-6
    assert abs(summary.min_h_obstacle - (-0.2)) < 1e-6
    assert abs(summary.avg_skill_switches - 5.0) < 1e-6
