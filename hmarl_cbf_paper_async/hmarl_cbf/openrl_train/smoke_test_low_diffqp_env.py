from __future__ import annotations

import numpy as np

from hmarl_cbf.openrl_envs import CoreEnv, LowLevelOpenRLEnv, LowLevelOpenRLEnvConfig
from hmarl_cbf.openrl_models import LowQPDecoder


def main() -> None:
    env_cfg = {
        "n_agents": 2,
        "n_obstacles": 2,
        "world_size": 6.0,
        "dt": 0.03,
        "horizon": 30,
        "action_limit": 1.0,
        "velocity_limit": 2.0,
        "agent_radius": 0.2,
        "goal_threshold": 0.4,
        "goal_speed_threshold": 0.3,
        "lidar_beams": 16,
        "lidar_range": 3.0,
        "lidar_noise_std": 0.0,
        "neighbor_radius": 3.0,
        "max_neighbors": 2,
        "obstacle_rect_prob": 0.5,
        "obstacle_circle_radius_min": 0.4,
        "obstacle_circle_radius_max": 1.0,
        "obstacle_rect_half_extent_min": 0.4,
        "obstacle_rect_half_extent_max": 1.0,
        "disturbance_enabled": False,
        "disturbance_accel_max": 0.0,
        "reward_progress_weight": 1.0,
        "reward_time_penalty": 0.003,
        "reward_reach_bonus": 2.0,
        "reward_collision_penalty": 2.0,
        "reward_oob_penalty": 2.0,
        "lidar_obstacle_cbf_enabled": True,
        "lidar_cbf_point_radius": 0.0,
        "lidar_cbf_top_k": 3,
    }

    core = CoreEnv.from_env_config(env_cfg)
    env = LowLevelOpenRLEnv(
        core,
        LowLevelOpenRLEnvConfig(
            n_skills=5,
            phi_dim=LowQPDecoder.compute_phi_dim(action_dim=2, parameterize_cbf_constraints=True),
            default_skill_id=2,
            d_min_agent=0.6,
            d_safe_obs=0.6,
            torch_device="cpu",
            decoder_config={"action_dim": 2, "parameterize_cbf_constraints": True},
        ),
    )

    obs, infos = env.reset(seed=0)
    actions = {aid: np.zeros((5,), dtype=np.float32) for aid in env.possible_agents}
    next_obs, rewards, terminations, truncations, step_infos = env.step(actions)

    agent0 = env.possible_agents[0]
    print("reset_agents", sorted(obs.keys()))
    print("next_agents", sorted(next_obs.keys()))
    print("reward_keys", sorted(rewards.keys()))
    print("adapter_mode", step_infos[agent0]["adapter_mode"])
    print("executed_action", np.asarray(step_infos[agent0]["executed_action"], dtype=np.float32).tolist())
    print("qp_slack_shape", np.asarray(step_infos[agent0]["qp_slack"], dtype=np.float32).shape)
    print("qp_cbf_slack_shape", np.asarray(step_infos[agent0]["qp_cbf_slack"], dtype=np.float32).shape)
    print("termination_flags", terminations)
    print("truncation_flags", truncations)


if __name__ == "__main__":
    main()
