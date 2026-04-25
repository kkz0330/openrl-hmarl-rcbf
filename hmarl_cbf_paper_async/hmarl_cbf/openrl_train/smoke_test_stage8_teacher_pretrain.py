from __future__ import annotations

from hmarl_cbf.openrl_agents import LowDiffQPAgent, LowDiffQPTeacherPretrainConfig
from hmarl_cbf.openrl_envs import CoreEnv, LowLevelOpenRLEnv, LowLevelOpenRLEnvConfig
from hmarl_cbf.openrl_models import LowQPActorNetwork, LowQPCriticNetwork, LowQPDecoder
from hmarl_cbf.openrl_train.pretrain_low_from_teacher import _low_cfg
from hmarl_cbf.openrl_train.teacher_dataset import (
    TeacherDatasetCollector,
    TeacherDatasetCollectorConfig,
    build_teacher_baseline_controller_from_config,
)
from hmarl_cbf.skills import build_default_skill_library


def _base_cfg():
    return {
        "env": {
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
        },
        "model": {"action_dim": 2},
        "safety": {"d_min_agent": 0.6, "d_safe_obs": 0.6, "boundary_cbf": True, "boundary_margin": 0.2},
        "low_level_qp": {
            "w_clf": 10.0,
            "w_cbf": 100.0,
            "cbf_slack_max": 1.0,
            "cbf_k0": 1.0,
            "cbf_k1": 1.0,
            "clf_k": 1.0,
            "hocbf_gamma_h": 1.0,
            "hocbf_gamma_hdot": 1.0,
            "f_residual_reference_enabled": True,
            "f_ref_speed": 1.2,
            "f_ref_kp": 1.2,
            "f_ref_slow_radius": 1.5,
            "f_ref_goal_stop_min_speed": 0.0,
        },
        "skills": {"params": {"cbf_mode": "distributed_gcbfplus", "cbf_share_agent": 0.5, "cbf_share_obs": 1.0, "ref_speed": 1.2, "slow_radius": 1.5, "goal_stop_min_speed": 0.0}},
        "teacher_baseline": {"nominal_mode": "lqr", "lqr_q_pos": 5.0, "lqr_q_vel": 5.0, "lqr_r_input": 1.0},
        "qp": {"use_stub_if_unavailable": False, "ecos_max_iters": 500, "scs_max_iters": 10000, "scs_eps": 1e-4},
        "teacher_pretrain": {"default_skill_id": 2, "hidden_dim": 128},
    }


def main() -> None:
    cfg = _base_cfg()
    core = CoreEnv.from_env_config(cfg["env"])
    n_skills = len(build_default_skill_library())
    low_env = LowLevelOpenRLEnv(
        core,
        LowLevelOpenRLEnvConfig(
            n_skills=n_skills,
            phi_dim=LowQPDecoder.compute_phi_dim(action_dim=2, parameterize_cbf_constraints=True),
            default_skill_id=2,
            d_min_agent=0.6,
            d_safe_obs=0.6,
            neighbor_perception_radius=3.0,
            obstacle_perception_range=3.0,
            torch_device="cpu",
            decoder_config={"action_dim": 2, "parameterize_cbf_constraints": True},
        ),
    )
    teacher = build_teacher_baseline_controller_from_config(cfg)
    dataset = TeacherDatasetCollector(
        low_env,
        teacher,
        TeacherDatasetCollectorConfig(
            episodes=1,
            max_steps_per_episode=4,
            seed=0,
            skill_selection_mode="cyclic",
        ),
    ).collect()

    cfg_openrl = _low_cfg(n_skills=n_skills)
    actor = LowQPActorNetwork(
        cfg_openrl,
        low_env.observation_space,
        low_env.action_space,
        device="cpu",
        extra_args={"n_skills": n_skills, "hidden_dim": 128},
    )
    critic = LowQPCriticNetwork(
        cfg_openrl,
        low_env.state_space,
        device="cpu",
        extra_args={"hidden_dim": 128},
    )
    decoder = LowQPDecoder(action_dim=2, parameterize_cbf_constraints=True)
    agent = LowDiffQPAgent(actor, critic, decoder, torch_device="cpu")
    agent.bind_env(low_env)
    metrics = agent.pretrain_from_teacher(
        dataset.samples,
        config=LowDiffQPTeacherPretrainConfig(epochs=1),
    )

    print("teacher_dataset_samples", int(len(dataset)))
    print("teacher_pretrain_n_updates", float(metrics["n_updates"]))
    print("teacher_pretrain_loss_action", float(metrics["loss_action"]))
    print("teacher_pretrain_loss_slack", float(metrics["loss_slack"]))
    print("teacher_pretrain_action_mae", float(metrics["teacher_action_mae"]))
    print("teacher_pretrain_feasible_rate", float(metrics["teacher_feasible_rate"]))


if __name__ == "__main__":
    main()
