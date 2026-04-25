from __future__ import annotations

from typing import Any

import numpy as np

from hmarl_cbf.openrl_compat import parse_openrl_default_config
from hmarl_cbf.openrl_agents import (
    HighMAPPOAgent,
    HighMAPPOAgentConfig,
    LowDiffQPAgent,
    LowDiffQPAgentConfig,
)
from hmarl_cbf.openrl_train.common import LowLevelPolicyExecutor
from hmarl_cbf.openrl_envs import (
    CoreEnv,
    HighLevelOpenRLEnv,
    HighLevelOpenRLEnvConfig,
    LowLevelOpenRLEnv,
    LowLevelOpenRLEnvConfig,
)
from hmarl_cbf.openrl_models import (
    HighLevelMAPPONet,
    LowQPActorNetwork,
    LowQPCriticNetwork,
    LowQPDecoder,
)


def _base_env_cfg() -> dict[str, Any]:
    return {
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


def _low_cfg(n_skills: int) -> Any:
    cfg = parse_openrl_default_config()
    cfg.n_skills = int(n_skills)
    cfg.use_naive_recurrent_policy = False
    cfg.use_recurrent_policy = False
    cfg.recurrent_N = 1
    cfg.use_fp16 = False
    cfg.use_deepspeed = False
    cfg.use_orthogonal = True
    cfg.rnn_type = "gru"
    cfg.gain = 0.01
    cfg.hidden_size = 128
    cfg.low_hidden_size = 128
    cfg.use_valuenorm = False
    cfg.use_policy_vhead = False
    return cfg


def run_high_level_smoke() -> None:
    core = CoreEnv.from_env_config(_base_env_cfg())
    phi_dim = LowQPDecoder.compute_phi_dim(action_dim=2, parameterize_cbf_constraints=True)
    low_env = LowLevelOpenRLEnv(
        core,
        LowLevelOpenRLEnvConfig(
            n_skills=5,
            phi_dim=phi_dim,
            default_skill_id=2,
            d_min_agent=0.6,
            d_safe_obs=0.6,
            torch_device="cpu",
            decoder_config={"action_dim": 2, "parameterize_cbf_constraints": True},
        ),
    )
    low_cfg = _low_cfg(n_skills=5)
    low_actor = LowQPActorNetwork(
        low_cfg,
        low_env.observation_space,
        low_env.action_space,
        device="cpu",
        extra_args={"n_skills": 5, "hidden_dim": 128},
    )
    low_critic = LowQPCriticNetwork(
        low_cfg,
        low_env.state_space,
        device="cpu",
        extra_args={"hidden_dim": 128},
    )
    low_decoder = LowQPDecoder(action_dim=2, parameterize_cbf_constraints=True)
    low_agent = LowDiffQPAgent(
        low_actor,
        low_critic,
        low_decoder,
        config=LowDiffQPAgentConfig(ppo_epochs=1, entropy_coef=0.0),
        torch_device="cpu",
    )
    low_agent.bind_env(low_env)
    low_executor = LowLevelPolicyExecutor(low_agent, n_skills=5)
    env = HighLevelOpenRLEnv(
        core,
        HighLevelOpenRLEnvConfig(
            n_skills=5,
            coordinator_mode="sync",
            t_sync_max=5,
            max_low_steps_per_high_step=4,
        ),
        low_level_executor=low_executor,
    )
    net = HighLevelMAPPONet(env=env, device="cpu", n_rollout_threads=1)
    net.reset()
    agent = HighMAPPOAgent(
        net,
        HighMAPPOAgentConfig(
            ppo_epochs=1,
            minibatch_size=8,
        ),
    )

    obs, infos = env.reset(seed=0)
    next_obs, rewards, terminations, truncations, next_infos, action_out = agent.step_env(
        env,
        obs,
        infos,
        deterministic=False,
    )
    metrics = agent.update()

    agent0 = env.possible_agents[0]
    print("high_reset_agents", sorted(obs.keys()))
    print("high_next_agents", sorted(next_obs.keys()))
    print("high_action", int(action_out[agent0]["action"]))
    print("high_reward", float(rewards[agent0]))
    print("high_switch_required_next", bool(next_infos[agent0]["switch_required_next"]))
    print("high_termination_flags", terminations)
    print("high_truncation_flags", truncations)
    print("high_update_n_samples", float(metrics["n_samples"]))
    print("high_update_loss_actor", float(metrics["loss_actor"]))
    print("high_update_loss_value", float(metrics["loss_value"]))


def run_low_level_smoke() -> None:
    core = CoreEnv.from_env_config(_base_env_cfg())
    phi_dim = LowQPDecoder.compute_phi_dim(action_dim=2, parameterize_cbf_constraints=True)
    env = LowLevelOpenRLEnv(
        core,
        LowLevelOpenRLEnvConfig(
            n_skills=5,
            phi_dim=phi_dim,
            default_skill_id=2,
            d_min_agent=0.6,
            d_safe_obs=0.6,
            torch_device="cpu",
            decoder_config={"action_dim": 2, "parameterize_cbf_constraints": True},
        ),
    )
    cfg = _low_cfg(n_skills=5)
    actor = LowQPActorNetwork(
        cfg,
        env.observation_space,
        env.action_space,
        device="cpu",
        extra_args={"n_skills": 5, "hidden_dim": 128},
    )
    critic = LowQPCriticNetwork(
        cfg,
        env.state_space,
        device="cpu",
        extra_args={"hidden_dim": 128},
    )
    decoder = LowQPDecoder(action_dim=2, parameterize_cbf_constraints=True)
    agent = LowDiffQPAgent(
        actor,
        critic,
        decoder,
        config=LowDiffQPAgentConfig(
            ppo_epochs=1,
            entropy_coef=0.0,
        ),
        torch_device="cpu",
    )

    obs, infos = env.reset(seed=1)
    next_obs, rewards, terminations, truncations, next_infos, action_out = agent.step_env(
        env,
        obs,
        infos,
        deterministic=False,
    )
    metrics = agent.update()

    agent0 = env.possible_agents[0]
    print("low_reset_agents", sorted(obs.keys()))
    print("low_next_agents", sorted(next_obs.keys()))
    print("low_phi_shape", np.asarray(action_out[agent0]["phi"], dtype=np.float32).shape)
    print("low_reward", float(rewards[agent0]))
    print("low_executed_action", np.asarray(next_infos[agent0]["executed_action"], dtype=np.float32).tolist())
    print("low_qp_slack_shape", np.asarray(next_infos[agent0]["qp_slack"], dtype=np.float32).shape)
    print("low_qp_cbf_slack_shape", np.asarray(next_infos[agent0]["qp_cbf_slack"], dtype=np.float32).shape)
    print("low_termination_flags", terminations)
    print("low_truncation_flags", truncations)
    print("low_update_n_updates", float(metrics["n_updates"]))
    print("low_update_loss_actor", float(metrics["loss_actor"]))
    print("low_update_loss_value", float(metrics["loss_value"]))
    print("low_update_loss_slack", float(metrics["loss_slack"]))


def main() -> None:
    run_high_level_smoke()
    run_low_level_smoke()


if __name__ == "__main__":
    main()
