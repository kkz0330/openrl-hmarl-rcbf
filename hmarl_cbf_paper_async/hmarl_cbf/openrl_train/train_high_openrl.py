from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from hmarl_cbf.openrl_agents import HighMAPPOAgent, HighMAPPOAgentConfig
from hmarl_cbf.openrl_train.common import (
    apply_config_section_defaults,
    LowLevelPolicyExecutor,
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


def train_high(cfg: dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    core = build_core_env(cfg)
    if not args.low_checkpoint:
        raise ValueError("train_high_openrl requires --low-checkpoint so high-level training uses a real low-level executor")
    low_env = build_low_env(cfg, core, torch_device=args.torch_device)
    low_agent = build_low_agent(low_env, cfg, torch_device=args.torch_device)
    low_agent.load(args.low_checkpoint)
    low_executor = LowLevelPolicyExecutor(low_agent, n_skills=len(build_default_skill_library()))

    env = build_high_env(cfg, core, low_level_executor=low_executor)
    net = build_high_net(env, torch_device=args.torch_device)
    net.reset()
    agent = HighMAPPOAgent(
        net,
        HighMAPPOAgentConfig(
            actor_lr=float(args.actor_lr),
            critic_lr=float(args.critic_lr),
            clip_ratio=float(args.clip_ratio),
            value_coef=float(args.value_coef),
            entropy_coef=float(args.entropy_coef),
            max_grad_norm=float(args.max_grad_norm),
            ppo_epochs=int(args.ppo_epochs),
            minibatch_size=int(args.minibatch_size),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            normalize_advantages=bool(args.normalize_advantages),
        ),
    )

    final_metrics: Dict[str, Any] = {}
    for episode in range(int(args.episodes)):
        if low_executor is not None:
            low_executor.reset(env.possible_agents)
        agent.reset()
        obs, infos = env.reset(seed=int(args.seed + episode))
        episode_reward = 0.0
        steps = 0
        while env.agents:
            obs, rewards, terminations, truncations, infos, _ = agent.step_env(
                env,
                obs,
                infos,
                deterministic=bool(args.deterministic_rollout),
            )
            episode_reward += float(sum(float(v) for v in rewards.values()))
            steps += 1
            if all(bool(terminations[aid] or truncations[aid]) for aid in terminations):
                break
        update_metrics = agent.update()
        row = {
            "episode": int(episode),
            "episode_reward_sum": float(episode_reward),
            "episode_high_steps": int(steps),
            **{str(k): float(v) for k, v in update_metrics.items()},
        }
        final_metrics = row
        print("high_train_episode", int(episode))
        print("high_train_reward_sum", float(episode_reward))
        print("high_train_steps", int(steps))
        print("high_train_loss_actor", float(update_metrics["loss_actor"]))
        print("high_train_loss_value", float(update_metrics["loss_value"]))
        if args.metrics_jsonl:
            append_jsonl(args.metrics_jsonl, row)
        if args.checkpoint_every > 0 and ((episode + 1) % int(args.checkpoint_every) == 0):
            ckpt = Path(args.checkpoint_dir) / f"high_episode_{episode + 1}.pt"
            agent.save(ckpt)

    final_ckpt = Path(args.checkpoint_dir) / "high_final.pt"
    agent.save(final_ckpt)
    final_metrics["checkpoint"] = str(final_ckpt)
    return final_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train high-level CTDE policy with OpenRL-style MAPPO agent")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--torch-device", type=str, default=None)
    parser.add_argument("--low-checkpoint", type=str, default=None)
    parser.add_argument("--deterministic-rollout", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--clip-ratio", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=None)
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
        section="openrl_high_train",
        defaults={
            "episodes": 50,
            "seed": 0,
            "torch_device": "cpu",
            "low_checkpoint": "",
            "deterministic_rollout": False,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "clip_ratio": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "ppo_epochs": 4,
            "minibatch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "normalize_advantages": False,
            "checkpoint_dir": "artifacts/openrl/high_run",
            "checkpoint_every": 0,
            "metrics_json": "",
            "metrics_jsonl": "",
        },
    )
    metrics = train_high(cfg, args)
    if args.metrics_json:
        save_json(args.metrics_json, metrics)


if __name__ == "__main__":
    main()
