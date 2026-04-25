from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from hmarl_cbf.openrl_agents import HighMAPPOAgent, HighMAPPOAgentConfig
from hmarl_cbf.openrl_train.common import (
    apply_config_section_defaults,
    JointLowLevelPolicyExecutor,
    append_jsonl,
    build_core_env,
    build_high_env,
    build_high_net,
    build_low_agent,
    build_low_env,
    load_yaml,
    save_json,
)
from hmarl_cbf.skills import build_default_skill_library


def train_joint(cfg: dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    core = build_core_env(cfg)
    low_env = build_low_env(cfg, core, torch_device=args.torch_device)
    low_agent = build_low_agent(low_env, cfg, torch_device=args.torch_device)
    if args.low_init_checkpoint:
        low_agent.load(args.low_init_checkpoint)

    joint_executor = JointLowLevelPolicyExecutor(
        low_agent,
        core,
        n_skills=len(build_default_skill_library()),
        include_local_low_obs_in_critic=bool(low_env.config.include_local_obs_in_critic),
        deterministic=bool(args.low_deterministic_rollout),
    )
    env = build_high_env(cfg, core, low_level_executor=joint_executor)

    net = build_high_net(env, torch_device=args.torch_device)
    net.reset()
    high_agent = HighMAPPOAgent(
        net,
        HighMAPPOAgentConfig(
            actor_lr=float(args.high_actor_lr),
            critic_lr=float(args.high_critic_lr),
            clip_ratio=float(args.high_clip_ratio),
            value_coef=float(args.high_value_coef),
            entropy_coef=float(args.high_entropy_coef),
            max_grad_norm=float(args.high_max_grad_norm),
            ppo_epochs=int(args.high_ppo_epochs),
            minibatch_size=int(args.high_minibatch_size),
            gamma=float(args.high_gamma),
            gae_lambda=float(args.high_gae_lambda),
            normalize_advantages=bool(args.high_normalize_advantages),
        ),
    )
    if args.high_init_checkpoint:
        high_agent.load(args.high_init_checkpoint)

    low_agent.config = type(low_agent.config)(
        actor_lr=float(args.low_actor_lr),
        critic_lr=float(args.low_critic_lr),
        clip_ratio=float(args.low_clip_ratio),
        value_coef=float(args.low_value_coef),
        entropy_coef=float(args.low_entropy_coef),
        max_grad_norm=float(args.low_max_grad_norm),
        ppo_epochs=int(args.low_ppo_epochs),
        gamma=float(args.low_gamma),
        gae_lambda=float(args.low_gae_lambda),
        normalize_advantages=bool(args.low_normalize_advantages),
        slack_coef=float(args.low_slack_coef),
        cbf_slack_coef=float(args.low_cbf_slack_coef),
        ext_reward_coef=float(args.low_ext_reward_coef),
        reward_mix_eta=float(args.low_reward_mix_eta),
        divide_high_adv_by_n_agents=bool(args.low_divide_high_adv_by_n_agents),
        reset_on_sync_switch=bool(args.low_reset_on_sync_switch),
        safety_margin_coef=float(args.low_safety_margin_coef),
        safety_margin_h_agent=float(args.low_safety_margin_h_agent),
        safety_margin_h_obstacle=float(args.low_safety_margin_h_obstacle),
    )

    final_metrics: Dict[str, Any] = {}
    max_high_steps = int(args.max_high_steps_per_episode)
    log_every = max(1, int(args.log_every))
    total_episodes = int(args.episodes)
    for episode in range(int(args.episodes)):
        joint_executor.reset(env.possible_agents)
        high_agent.reset()
        obs, infos = env.reset(seed=int(args.seed + episode))
        episode_reward = 0.0
        high_steps = 0
        qp_count = 0
        qp_feasible_count = 0
        qp_fallback_count = 0

        while env.agents:
            obs, rewards, terminations, truncations, infos, _ = high_agent.step_env(
                env,
                obs,
                infos,
                deterministic=bool(args.high_deterministic_rollout),
            )
            episode_reward += float(sum(float(v) for v in rewards.values()))
            high_steps += 1
            for step_record in env.last_low_step_records:
                adapter_info = dict(step_record.get("adapter_info", {}))
                for item in adapter_info.values():
                    info = dict(item or {})
                    if not info:
                        continue
                    qp_count += 1
                    qp_feasible_count += int(bool(info.get("qp_feasible", False)))
                    qp_fallback_count += int(bool(info.get("qp_used_fallback", False)))
            if max_high_steps > 0 and high_steps >= max_high_steps:
                break
            if not env.agents or all(bool(terminations[aid] or truncations[aid]) for aid in terminations):
                break

        high_adv_by_option = high_agent.advantages_by_option()
        low_update = low_agent.update(
            high_adv_by_option=high_adv_by_option,
            n_agents=len(env.possible_agents),
        )
        high_update = high_agent.update()
        row = {
            "episode": int(episode),
            "episode_reward_sum": float(episode_reward),
            "episode_high_steps": int(high_steps),
            "episode_qp_count": int(qp_count),
            "episode_qp_feasible_count": int(qp_feasible_count),
            "episode_qp_feasible_rate": float(qp_feasible_count / max(1, qp_count)),
            "episode_qp_fallback_count": int(qp_fallback_count),
            "episode_qp_fallback_rate": float(qp_fallback_count / max(1, qp_count)),
            **{f"low_{k}": float(v) for k, v in low_update.items()},
            **{f"high_{k}": float(v) for k, v in high_update.items()},
        }
        final_metrics = row
        should_log = (episode == 0) or ((episode + 1) % log_every == 0) or ((episode + 1) == total_episodes)
        if should_log:
            print("joint_train_episode", int(episode + 1))
            print("joint_train_reward_sum", float(episode_reward))
            print("joint_train_high_steps", int(high_steps))
            print("joint_train_qp_count", int(qp_count))
            print("joint_train_qp_feasible_rate", float(qp_feasible_count / max(1, qp_count)))
            print("joint_train_qp_fallback_rate", float(qp_fallback_count / max(1, qp_count)))
            print("joint_train_low_loss_actor", float(low_update["loss_actor"]))
            print("joint_train_low_loss_value", float(low_update["loss_value"]))
            if "loss_slack" in low_update:
                print("joint_train_low_loss_slack", float(low_update["loss_slack"]))
            if "loss_cbf_slack" in low_update:
                print("joint_train_low_loss_cbf_slack", float(low_update["loss_cbf_slack"]))
            print("joint_train_high_loss_actor", float(high_update["loss_actor"]))
            print("joint_train_high_loss_value", float(high_update["loss_value"]))
        if args.metrics_jsonl:
            append_jsonl(args.metrics_jsonl, row)
        if args.checkpoint_every > 0 and ((episode + 1) % int(args.checkpoint_every) == 0):
            ckpt_dir = Path(args.checkpoint_dir)
            low_agent.save(ckpt_dir / f"low_episode_{episode + 1}.pt")
            high_agent.save(ckpt_dir / f"high_episode_{episode + 1}.pt")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    low_final = ckpt_dir / "low_final.pt"
    high_final = ckpt_dir / "high_final.pt"
    low_agent.save(low_final)
    high_agent.save(high_final)
    final_metrics["low_checkpoint"] = str(low_final)
    final_metrics["high_checkpoint"] = str(high_final)
    return final_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint synchronized high/low OpenRL training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--torch-device", type=str, default=None)
    parser.add_argument("--low-init-checkpoint", type=str, default=None)
    parser.add_argument("--high-init-checkpoint", type=str, default=None)
    parser.add_argument("--max-high-steps-per-episode", type=int, default=None)
    parser.add_argument("--low-deterministic-rollout", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--high-deterministic-rollout", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--low-actor-lr", type=float, default=None)
    parser.add_argument("--low-critic-lr", type=float, default=None)
    parser.add_argument("--low-clip-ratio", type=float, default=None)
    parser.add_argument("--low-value-coef", type=float, default=None)
    parser.add_argument("--low-entropy-coef", type=float, default=None)
    parser.add_argument("--low-max-grad-norm", type=float, default=None)
    parser.add_argument("--low-ppo-epochs", type=int, default=None)
    parser.add_argument("--low-gamma", type=float, default=None)
    parser.add_argument("--low-gae-lambda", type=float, default=None)
    parser.add_argument("--low-normalize-advantages", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--low-slack-coef", type=float, default=None)
    parser.add_argument("--low-cbf-slack-coef", type=float, default=None)
    parser.add_argument("--low-ext-reward-coef", type=float, default=None)
    parser.add_argument("--low-reward-mix-eta", type=float, default=None)
    parser.add_argument("--low-divide-high-adv-by-n-agents", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--low-reset-on-sync-switch", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--low-safety-margin-coef", type=float, default=None)
    parser.add_argument("--low-safety-margin-h-agent", type=float, default=None)
    parser.add_argument("--low-safety-margin-h-obstacle", type=float, default=None)

    parser.add_argument("--high-actor-lr", type=float, default=None)
    parser.add_argument("--high-critic-lr", type=float, default=None)
    parser.add_argument("--high-clip-ratio", type=float, default=None)
    parser.add_argument("--high-value-coef", type=float, default=None)
    parser.add_argument("--high-entropy-coef", type=float, default=None)
    parser.add_argument("--high-max-grad-norm", type=float, default=None)
    parser.add_argument("--high-ppo-epochs", type=int, default=None)
    parser.add_argument("--high-minibatch-size", type=int, default=None)
    parser.add_argument("--high-gamma", type=float, default=None)
    parser.add_argument("--high-gae-lambda", type=float, default=None)
    parser.add_argument("--high-normalize-advantages", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument("--metrics-jsonl", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    args = apply_config_section_defaults(
        args,
        cfg,
        section="openrl_joint_train",
        defaults={
            "episodes": 50,
            "seed": 0,
            "torch_device": "cpu",
            "low_init_checkpoint": "",
            "high_init_checkpoint": "",
            "max_high_steps_per_episode": 0,
            "low_deterministic_rollout": False,
            "high_deterministic_rollout": False,
            "low_actor_lr": 3e-4,
            "low_critic_lr": 3e-4,
            "low_clip_ratio": 0.2,
            "low_value_coef": 0.5,
            "low_entropy_coef": 0.0,
            "low_max_grad_norm": 0.5,
            "low_ppo_epochs": 2,
            "low_gamma": 0.99,
            "low_gae_lambda": 0.95,
            "low_normalize_advantages": False,
            "low_slack_coef": 0.02,
            "low_cbf_slack_coef": 0.05,
            "low_ext_reward_coef": 0.0,
            "low_reward_mix_eta": 0.35,
            "low_divide_high_adv_by_n_agents": True,
            "low_reset_on_sync_switch": True,
            "low_safety_margin_coef": 0.0,
            "low_safety_margin_h_agent": 0.0,
            "low_safety_margin_h_obstacle": 0.0,
            "high_actor_lr": 3e-4,
            "high_critic_lr": 3e-4,
            "high_clip_ratio": 0.2,
            "high_value_coef": 0.5,
            "high_entropy_coef": 0.01,
            "high_max_grad_norm": 0.5,
            "high_ppo_epochs": 4,
            "high_minibatch_size": 64,
            "high_gamma": 0.99,
            "high_gae_lambda": 0.95,
            "high_normalize_advantages": False,
            "checkpoint_dir": "artifacts/openrl/joint_run",
            "checkpoint_every": 0,
            "log_every": 20,
            "metrics_json": "",
            "metrics_jsonl": "",
        },
    )
    metrics = train_joint(cfg, args)
    if args.metrics_json:
        save_json(args.metrics_json, metrics)


if __name__ == "__main__":
    main()
