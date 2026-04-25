from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from hmarl_cbf.openrl_train.common import load_yaml, save_json
from hmarl_cbf.openrl_train.common import apply_config_section_defaults
from hmarl_cbf.openrl_train.pretrain_low_from_teacher import (
    _build_core_env,
    _build_low_agent,
    _build_low_env,
)
from hmarl_cbf.openrl_train.teacher_dataset import (
    TeacherDatasetCollector,
    TeacherDatasetCollectorConfig,
    build_teacher_controller_from_config,
)
from hmarl_cbf.openrl_agents import (
    HighMAPPOAgent,
    HighMAPPOAgentConfig,
    HighMAPPOTeacherPretrainConfig,
    LowDiffQPTeacherPretrainConfig,
)
from hmarl_cbf.openrl_train.train_joint_openrl import train_joint
from hmarl_cbf.openrl_train.common import build_high_env, build_high_net


def train_stack(cfg: dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    low_init_checkpoint = ""
    high_init_checkpoint = ""
    teacher_metrics: Dict[str, Any] = {}
    if not args.skip_teacher_warmstart:
        dataset_path = artifacts_dir / "teacher_low_dataset.pkl"
        warmstart_ckpt = artifacts_dir / "low_teacher_warmstart.pt"
        high_warmstart_ckpt = artifacts_dir / "high_teacher_warmstart.pt"
        print("stack_teacher_source", str(args.teacher_source))
        if str(args.teacher_checkpoint).strip():
            print("stack_teacher_checkpoint", str(args.teacher_checkpoint))
        print("stack_teacher_collect_start", int(args.teacher_episodes))
        print("stack_teacher_collect_max_steps", int(args.teacher_max_steps_per_episode))
        core = _build_core_env(cfg)
        low_env = _build_low_env(cfg, core, torch_device=args.torch_device)
        teacher = build_teacher_controller_from_config(
            cfg,
            teacher_source=str(args.teacher_source),
            teacher_checkpoint=str(args.teacher_checkpoint),
            torch_device=str(args.torch_device),
            deterministic=True,
        )
        collector = TeacherDatasetCollector(
            low_env,
            teacher,
            TeacherDatasetCollectorConfig(
                episodes=int(args.teacher_episodes),
                max_steps_per_episode=int(args.teacher_max_steps_per_episode),
                seed=int(args.seed),
                skill_selection_mode=str(args.teacher_skill_selection_mode),
            ),
        )
        dataset = collector.collect()
        dataset.save(dataset_path)
        print("stack_teacher_high_samples", int(len(collector.last_high_level_samples)))
        if collector.last_high_level_samples:
            print("stack_teacher_high_pretrain_start", int(args.teacher_high_pretrain_epochs))
            high_env = build_high_env(cfg, core, low_level_executor=low_env.action_adapter)
            high_net = build_high_net(high_env, torch_device=args.torch_device)
            high_net.reset()
            high_agent = HighMAPPOAgent(
                high_net,
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
            high_teacher_metrics = high_agent.pretrain_from_teacher(
                collector.last_high_level_samples,
                config=HighMAPPOTeacherPretrainConfig(
                    epochs=int(args.teacher_high_pretrain_epochs),
                    actor_lr=float(args.teacher_high_actor_lr),
                    entropy_coef=float(args.teacher_high_entropy_coef),
                    max_grad_norm=float(args.teacher_high_max_grad_norm),
                ),
            )
            high_agent.save(high_warmstart_ckpt)
            high_init_checkpoint = str(high_warmstart_ckpt)
            teacher_metrics["high_teacher_pretrain"] = high_teacher_metrics
            print("stack_teacher_high_warmstart_checkpoint", str(high_warmstart_ckpt))
        print("stack_teacher_pretrain_start", int(args.teacher_pretrain_epochs))
        low_agent = _build_low_agent(low_env, cfg, torch_device=args.torch_device)
        low_teacher_metrics = low_agent.pretrain_from_teacher(
            dataset.samples,
            config=LowDiffQPTeacherPretrainConfig(
                epochs=int(args.teacher_pretrain_epochs),
                slack_coef=float(args.teacher_slack_coef),
                cbf_slack_coef=float(args.teacher_cbf_slack_coef),
                entropy_coef=float(args.teacher_entropy_coef),
                max_grad_norm=float(args.teacher_max_grad_norm),
            ),
        )
        low_agent.save(warmstart_ckpt)
        low_init_checkpoint = str(warmstart_ckpt)
        teacher_metrics["low_teacher_pretrain"] = low_teacher_metrics
        print("stack_teacher_dataset", str(dataset_path))
        print("stack_teacher_samples", int(len(dataset)))
        print("stack_teacher_warmstart_checkpoint", str(warmstart_ckpt))

    joint_args = argparse.Namespace(
        config=args.config,
        episodes=int(args.high_episodes),
        seed=int(args.seed),
        torch_device=args.torch_device,
        low_init_checkpoint=low_init_checkpoint,
        high_init_checkpoint=high_init_checkpoint,
        max_high_steps_per_episode=0,
        low_deterministic_rollout=bool(args.low_deterministic_rollout),
        high_deterministic_rollout=bool(args.high_deterministic_rollout),
        low_actor_lr=float(args.low_actor_lr),
        low_critic_lr=float(args.low_critic_lr),
        low_clip_ratio=float(args.low_clip_ratio),
        low_value_coef=float(args.low_value_coef),
        low_entropy_coef=float(args.low_entropy_coef),
        low_max_grad_norm=float(args.low_max_grad_norm),
        low_ppo_epochs=int(args.low_ppo_epochs),
        low_gamma=float(args.low_gamma),
        low_gae_lambda=float(args.low_gae_lambda),
        low_normalize_advantages=bool(args.low_normalize_advantages),
        low_slack_coef=float(args.low_slack_coef),
        low_cbf_slack_coef=float(args.low_cbf_slack_coef),
        low_ext_reward_coef=float(args.low_ext_reward_coef),
        low_reward_mix_eta=float(args.low_reward_mix_eta),
        low_divide_high_adv_by_n_agents=bool(args.low_divide_high_adv_by_n_agents),
        low_reset_on_sync_switch=bool(args.low_reset_on_sync_switch),
        low_safety_margin_coef=float(args.low_safety_margin_coef),
        low_safety_margin_h_agent=float(args.low_safety_margin_h_agent),
        low_safety_margin_h_obstacle=float(args.low_safety_margin_h_obstacle),
        high_actor_lr=float(args.high_actor_lr),
        high_critic_lr=float(args.high_critic_lr),
        high_clip_ratio=float(args.high_clip_ratio),
        high_value_coef=float(args.high_value_coef),
        high_entropy_coef=float(args.high_entropy_coef),
        high_max_grad_norm=float(args.high_max_grad_norm),
        high_ppo_epochs=int(args.high_ppo_epochs),
        high_minibatch_size=int(args.high_minibatch_size),
        high_gamma=float(args.high_gamma),
        high_gae_lambda=float(args.high_gae_lambda),
        high_normalize_advantages=bool(args.high_normalize_advantages),
        checkpoint_dir=str(artifacts_dir / "joint"),
        checkpoint_every=int(args.high_checkpoint_every),
        log_every=int(args.log_every),
        metrics_json="",
        metrics_jsonl=str(artifacts_dir / "joint_metrics.jsonl"),
    )
    joint_metrics = train_joint(cfg, joint_args)

    final = {
        "teacher_metrics": teacher_metrics,
        "joint_metrics": joint_metrics,
        "low_checkpoint": low_init_checkpoint if low_init_checkpoint else str(Path(joint_args.checkpoint_dir) / "low_final.pt"),
        "high_checkpoint": str(Path(joint_args.checkpoint_dir) / "high_final.pt"),
    }
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential stack training for rebuilt OpenRL HMARL-CBF")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--torch-device", type=str, default=None)

    parser.add_argument("--skip-teacher-warmstart", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--teacher-source", type=str, default=None, choices=["baseline", "model_checkpoint"])
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--teacher-episodes", type=int, default=None)
    parser.add_argument("--teacher-max-steps-per-episode", type=int, default=None)
    parser.add_argument("--teacher-skill-selection-mode", type=str, default=None, choices=["cyclic", "random"])
    parser.add_argument("--teacher-pretrain-epochs", type=int, default=None)
    parser.add_argument("--teacher-high-pretrain-epochs", type=int, default=None)
    parser.add_argument("--teacher-high-actor-lr", type=float, default=None)
    parser.add_argument("--teacher-high-entropy-coef", type=float, default=None)
    parser.add_argument("--teacher-high-max-grad-norm", type=float, default=None)
    parser.add_argument("--teacher-slack-coef", type=float, default=None)
    parser.add_argument("--teacher-cbf-slack-coef", type=float, default=None)
    parser.add_argument("--teacher-entropy-coef", type=float, default=None)
    parser.add_argument("--teacher-max-grad-norm", type=float, default=None)

    parser.add_argument("--low-deterministic-rollout", action=argparse.BooleanOptionalAction, default=None)
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

    parser.add_argument("--high-episodes", type=int, default=None)
    parser.add_argument("--high-deterministic-rollout", action=argparse.BooleanOptionalAction, default=None)
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
    parser.add_argument("--high-checkpoint-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)

    parser.add_argument("--metrics-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    args = apply_config_section_defaults(
        args,
        cfg,
        section="openrl_stack_train",
        defaults={
            "seed": 0,
            "torch_device": "cpu",
            "artifacts_dir": "artifacts/openrl/stack_run",
            "skip_teacher_warmstart": False,
            "teacher_source": "model_checkpoint",
            "teacher_checkpoint": "artifacts/hmarl_cbf_paper_async/phi_ppo_async_trapaware_stuck_rcbf_lidarpointcbf_topk_hardresume_round2/checkpoints/last.pt",
            "teacher_episodes": 8,
            "teacher_max_steps_per_episode": 0,
            "teacher_skill_selection_mode": "cyclic",
            "teacher_pretrain_epochs": 5,
            "teacher_high_pretrain_epochs": 5,
            "teacher_high_actor_lr": 3e-4,
            "teacher_high_entropy_coef": 0.0,
            "teacher_high_max_grad_norm": 0.5,
            "teacher_slack_coef": 0.02,
            "teacher_cbf_slack_coef": 0.05,
            "teacher_entropy_coef": 0.0,
            "teacher_max_grad_norm": 0.5,
            "low_deterministic_rollout": False,
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
            "high_episodes": 50,
            "high_deterministic_rollout": False,
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
            "high_checkpoint_every": 0,
            "log_every": 20,
            "metrics_json": "",
        },
    )
    metrics = train_stack(cfg, args)
    if args.metrics_json:
        save_json(args.metrics_json, metrics)


if __name__ == "__main__":
    main()
