import numpy as np

from hmarl_cbf.env import MultiUAV2DEnv


def test_env_collision_cost_and_termination() -> None:
    env = MultiUAV2DEnv(n_agents=2, n_obstacles=0, horizon=20, terminate_on_collision=True)
    _, info = env.reset(
        seed=0,
        options={
            "states": [
                {"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [2.0, 0.0]},
                {"position": [0.1, 0.0], "velocity": [0.0, 0.0], "goal": [-2.0, 0.0]},
            ]
        },
    )
    assert info["collision_flags"][0] is True
    assert info["unsafe_flags"][1] is True
    assert info["costs"][0] == 1.0

    _, _, terminated, truncated, step_info = env.step({0: np.zeros(2), 1: np.zeros(2)})
    assert terminated is True
    assert truncated is False
    assert step_info["unsafe_flags"][0] is True


def test_env_reach_mask() -> None:
    env = MultiUAV2DEnv(n_agents=2, n_obstacles=0, horizon=20, terminate_on_collision=False)
    _, info = env.reset(
        seed=1,
        options={
            "states": [
                {"position": [1.0, 1.0], "velocity": [0.0, 0.0], "goal": [1.0, 1.0]},
                {"position": [-1.0, -1.0], "velocity": [0.0, 0.0], "goal": [-1.0, -1.0]},
            ]
        },
    )
    assert info["reach_flags"][0] is True
    assert info["reach_flags"][1] is True
    _, _, terminated, _, step_info = env.step({0: np.zeros(2), 1: np.zeros(2)})
    assert terminated is True
    assert step_info["reach_flags"][0] is True


def test_env_sampling_non_overlapping_starts() -> None:
    env = MultiUAV2DEnv(
        n_agents=4,
        n_obstacles=2,
        min_agent_separation=0.9,
        min_start_goal_separation=1.2,
    )
    env.reset(seed=7)
    states = env.get_agent_states()
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            dist = np.linalg.norm(states[i].position - states[j].position)
            assert dist >= 0.9
