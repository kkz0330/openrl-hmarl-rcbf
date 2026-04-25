from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from hmarl_cbf.openrl_agents import (
    LowDiffQPAgent,
    LowDiffQPTeacherPretrainConfig,
)
from hmarl_cbf.openrl_compat import parse_openrl_default_config
from hmarl_cbf.openrl_envs import CoreEnv, LowLevelOpenRLEnv, LowLevelOpenRLEnvConfig
from hmarl_cbf.openrl_models import LowQPActorNetwork, LowQPCriticNetwork, LowQPDecoder
from hmarl_cbf.openrl_train.teacher_dataset import (
    LowLevelTeacherDataset,
    TeacherDatasetCollector,
    TeacherDatasetCollectorConfig,
    build_teacher_controller_from_config,
)
from hmarl_cbf.openrl_train.common import build_low_decoder_kwargs, infer_low_phi_dim
from hmarl_cbf.skills import build_default_skill_library


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict):
        raise ValueError("config root must be a mapping")
    return payload


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


def _build_core_env(cfg: Mapping[str, Any]) -> CoreEnv:
    return CoreEnv.from_env_config(cfg["env"])


def _build_low_env(cfg: Mapping[str, Any], core: CoreEnv, *, torch_device: str) -> LowLevelOpenRLEnv:
    phi_dim = infer_low_phi_dim(cfg)
    n_skills = len(build_default_skill_library())
    low_cfg = LowLevelOpenRLEnvConfig(
        n_skills=n_skills,
        phi_dim=phi_dim,
        default_skill_id=int(cfg.get("teacher_pretrain", {}).get("default_skill_id", 2)),
        d_min_agent=float(cfg["safety"]["d_min_agent"]),
        d_safe_obs=float(cfg["safety"]["d_safe_obs"]),
        neighbor_perception_radius=float(cfg["env"].get("neighbor_radius", 0.0)),
        obstacle_perception_range=float(cfg["env"].get("lidar_range", 0.0)),
        include_local_obs_in_critic=bool(cfg.get("teacher_pretrain", {}).get("include_local_obs_in_critic", False)),
        reactivate_same_skill_on_switch=False,
        torch_device=torch_device,
        decoder_config=build_low_decoder_kwargs(cfg),
    )
    return LowLevelOpenRLEnv(core, low_cfg)


def _build_low_agent(env: LowLevelOpenRLEnv, cfg: Mapping[str, Any], *, torch_device: str) -> LowDiffQPAgent:
    n_skills = len(build_default_skill_library())
    model_cfg = _low_cfg(n_skills=n_skills)
    hidden_dim = int(cfg.get("teacher_pretrain", {}).get("hidden_dim", 128))
    decoder_kwargs = build_low_decoder_kwargs(cfg)

    actor = LowQPActorNetwork(
        model_cfg,
        env.observation_space,
        env.action_space,
        device=torch_device,
        extra_args={"n_skills": n_skills, "hidden_dim": hidden_dim},
    )
    critic = LowQPCriticNetwork(
        model_cfg,
        env.state_space,
        device=torch_device,
        extra_args={"hidden_dim": hidden_dim},
    )
    decoder = LowQPDecoder(
        **decoder_kwargs,
    )
    agent = LowDiffQPAgent(actor, critic, decoder, torch_device=torch_device)
    agent.bind_env(env)
    return agent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm-start low-level diff-QP policy from teacher rollouts")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--metrics-path", type=str, default="")
    parser.add_argument("--teacher-source", type=str, default="baseline", choices=["baseline", "model_checkpoint"])
    parser.add_argument("--teacher-checkpoint", type=str, default="")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--force-collect", action="store_true")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps-per-episode", type=int, default=0)
    parser.add_argument("--skill-selection-mode", type=str, default="cyclic", choices=["cyclic", "random"])
    parser.add_argument("--teacher-feasible-only", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_yaml(args.config)
    core = _build_core_env(cfg)
    low_env = _build_low_env(cfg, core, torch_device=args.torch_device)
    teacher = build_teacher_controller_from_config(
        cfg,
        teacher_source=str(args.teacher_source),
        teacher_checkpoint=str(args.teacher_checkpoint),
        torch_device=str(args.torch_device),
        deterministic=True,
    )
    dataset_path = Path(args.dataset_path)

    if args.force_collect or not dataset_path.exists():
        collector = TeacherDatasetCollector(
            low_env,
            teacher,
            TeacherDatasetCollectorConfig(
                episodes=int(args.episodes),
                max_steps_per_episode=int(args.max_steps_per_episode),
                seed=int(args.seed),
                skill_selection_mode=str(args.skill_selection_mode),
                require_teacher_feasible=bool(args.teacher_feasible_only),
            ),
        )
        dataset = collector.collect()
        dataset.save(dataset_path)
        print("teacher_dataset_saved", str(dataset_path))
        print("teacher_dataset_samples", int(len(dataset)))
    else:
        dataset = LowLevelTeacherDataset.load(dataset_path)
        print("teacher_dataset_loaded", str(dataset_path))
        print("teacher_dataset_samples", int(len(dataset)))

    if args.collect_only:
        return

    agent = _build_low_agent(low_env, cfg, torch_device=args.torch_device)
    metrics = agent.pretrain_from_teacher(
        dataset.samples,
        config=LowDiffQPTeacherPretrainConfig(
            epochs=int(args.pretrain_epochs),
            action_coef=1.0,
            slack_coef=float(cfg.get("teacher_pretrain", {}).get("slack_coef", 0.02)),
            cbf_slack_coef=float(cfg.get("teacher_pretrain", {}).get("cbf_slack_coef", 0.05)),
            entropy_coef=float(cfg.get("teacher_pretrain", {}).get("entropy_coef", 0.0)),
            max_grad_norm=float(cfg.get("teacher_pretrain", {}).get("max_grad_norm", 0.5)),
        ),
    )
    checkpoint_path = Path(args.checkpoint_path)
    agent.save(checkpoint_path)
    print("teacher_pretrain_checkpoint", str(checkpoint_path))
    for key, value in metrics.items():
        print(f"teacher_pretrain_{key}", float(value))

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
