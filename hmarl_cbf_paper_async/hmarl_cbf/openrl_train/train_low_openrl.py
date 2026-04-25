from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from hmarl_cbf.openrl_agents import LowDiffQPAgentConfig
from hmarl_cbf.openrl_train.common import (
    apply_config_section_defaults,
    CyclicSkillScheduler,
    append_jsonl,
    build_core_env,
    build_low_agent,
    build_low_env,
    load_yaml,
    save_json,
)


def train_low(cfg: dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    core = build_core_env(cfg)
    env = build_low_env(cfg, core, torch_device=args.torch_device)
    agent = build_low_agent(env, cfg, torch_device=args.torch_device)
    if args.init_checkpoint:
        agent.load(args.init_checkpoint)
    if args.checkpoint_after_warmstart:
        agent.save(args.checkpoint_after_warmstart)

    agent.config = LowDiffQPAgentConfig(
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        clip_ratio=float(args.clip_ratio),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        max_grad_norm=float(args.max_grad_norm),
        ppo_epochs=int(args.ppo_epochs),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        normalize_advantages=bool(args.normalize_advantages),
        slack_coef=float(args.slack_coef),
        cbf_slack_coef=float(args.cbf_slack_coef),
    )

    scheduler = CyclicSkillScheduler(mode=str(args.skill_selection_mode), seed=int(args.seed))
    final_metrics: Dict[str, Any] = {}
    for episode in range(int(args.episodes)):
        agent.reset()
        obs, infos = env.reset(seed=int(args.seed + episode))
        episode_reward = 0.0
        steps = 0
        while True:
            scheduler.assign_pending(env)
            if not env.agents:
                break
            obs, rewards, terminations, truncations, infos, _ = agent.step_env(
                env,
                obs,
                infos,
                deterministic=bool(args.deterministic_rollout),
            )
            episode_reward += float(sum(float(v) for v in rewards.values()))
            steps += 1
            if not env.agents or all(bool(terminations[aid] or truncations[aid]) for aid in terminations):
                break
        update_metrics = agent.update()
        row = {
            "episode": int(episode),
            "episode_reward_sum": float(episode_reward),
            "episode_low_steps": int(steps),
            **{str(k): float(v) for k, v in update_metrics.items()},
        }
        final_metrics = row
        print("low_train_episode", int(episode))
        print("low_train_reward_sum", float(episode_reward))
        print("low_train_steps", int(steps))
        print("low_train_loss_actor", float(update_metrics["loss_actor"]))
        print("low_train_loss_value", float(update_metrics["loss_value"]))
        if args.metrics_jsonl:
            append_jsonl(args.metrics_jsonl, row)
        if args.checkpoint_every > 0 and ((episode + 1) % int(args.checkpoint_every) == 0):
            ckpt = Path(args.checkpoint_dir) / f"low_episode_{episode + 1}.pt"
            agent.save(ckpt)

    final_ckpt = Path(args.checkpoint_dir) / "low_final.pt"
    agent.save(final_ckpt)
    final_metrics["checkpoint"] = str(final_ckpt)
    return final_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train low-level diff-QP policy with OpenRL-style PPO agent")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--torch-device", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-after-warmstart", type=str, default=None)
    parser.add_argument("--skill-selection-mode", type=str, default=None, choices=["cyclic", "random"])
    parser.add_argument("--deterministic-rollout", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--clip-ratio", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--slack-coef", type=float, default=None)
    parser.add_argument("--cbf-slack-coef", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument("--metrics-jsonl", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    args = apply_config_section_defaults(
        args,
        cfg,
        section="openrl_low_train",
        defaults={
            "episodes": 100,
            "seed": 0,
            "torch_device": "cpu",
            "init_checkpoint": "",
            "checkpoint_after_warmstart": "",
            "skill_selection_mode": "cyclic",
            "deterministic_rollout": False,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "clip_ratio": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.0,
            "max_grad_norm": 0.5,
            "ppo_epochs": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "normalize_advantages": False,
            "slack_coef": 0.02,
            "cbf_slack_coef": 0.05,
            "checkpoint_dir": "artifacts/openrl/low_run",
            "checkpoint_every": 0,
            "metrics_json": "",
            "metrics_jsonl": "",
        },
    )
    metrics = train_low(cfg, args)
    if args.metrics_json:
        save_json(args.metrics_json, metrics)


if __name__ == "__main__":
    main()
