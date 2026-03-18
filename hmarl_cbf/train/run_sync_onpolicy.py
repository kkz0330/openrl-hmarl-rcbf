from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime entrypoint
    raise RuntimeError("PyTorch is required to run training") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime entrypoint
    raise RuntimeError("PyYAML is required to load training config") from exc

from hmarl_cbf.buffer import HierRolloutBuffer
from hmarl_cbf.control import (
    ConstraintBuilder,
    DifferentiableQPSolver,
    LowLevelSafeController,
    SyncCoordinator,
    TorchDifferentiableQPSolver,
)
from hmarl_cbf.env import MultiUAV2DEnv
from hmarl_cbf.high_level import MAPPOConfig, OnPolicyMAPPO
from hmarl_cbf.policies import HighLevelPolicy, LowLevelQPPolicy
from hmarl_cbf.skills import SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.train import TrainerHooks, TrainerSyncOnPolicy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synchronous HMARL-CBF and export final evaluation video.")
    parser.add_argument("--config", type=str, default="configs/hmarl_cbf/default_sync_onpolicy.yaml")
    parser.add_argument("--output-root", type=str, default="artifacts/hmarl_cbf")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--total-iterations", type=int, default=-1)
    parser.add_argument("--eval-interval", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--eval-episodes", type=int, default=1, help="Periodic evaluation episodes during training.")
    parser.add_argument("--final-eval-episodes", type=int, default=3, help="Final evaluation episodes with rendering.")
    parser.add_argument("--final-render-gif", action="store_true", help="Render final evaluation as GIF files.")
    parser.add_argument("--final-render-png", action="store_true", help="Render final evaluation as PNG files.")
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument(
        "--warmstart-low-hfg-from",
        type=str,
        default="",
        help="Checkpoint path for low-level H/F/gamma warm start.",
    )
    parser.add_argument(
        "--warmstart-include-backbone",
        action="store_true",
        help="Also warm start low-level shared encoder/embedding/fusion backbone.",
    )
    return parser.parse_args()


def _make_run_dir(output_root: Path, run_name: str) -> Path:
    if run_name:
        run_dir = output_root / run_name
    else:
        run_dir = output_root / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_media").mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_history_csv(path: Path, history: list[Dict[str, float]]) -> None:
    if not history:
        return
    keys: list[str] = []
    for row in history:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _extract_low_policy_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    low = ckpt.get("low_policy")
    if isinstance(low, dict):
        return low
    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt  # raw state_dict fallback
    raise ValueError("checkpoint does not contain low_policy state_dict")


def _warmstart_low_policy_hfg(
    low_policy: LowLevelQPPolicy,
    checkpoint_path: Path,
    include_backbone: bool,
) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"warmstart checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"invalid checkpoint format: {checkpoint_path}")
    source = _extract_low_policy_state_dict(ckpt)
    target = low_policy.state_dict()

    prefixes = [
        "r_diag_head.",
    ]
    if include_backbone:
        prefixes.extend(
            [
                "skill_embedding.",
                "obs_encoder.",
                "fusion.",
            ]
        )

    loaded: list[str] = []
    missing: list[str] = []
    shape_mismatch: list[str] = []
    for key in sorted(target.keys()):
        if not any(key.startswith(p) for p in prefixes):
            continue
        if key not in source:
            missing.append(key)
            continue
        if tuple(target[key].shape) != tuple(source[key].shape):
            shape_mismatch.append(key)
            continue
        target[key] = source[key]
        loaded.append(key)
    low_policy.load_state_dict(target, strict=True)
    return {
        "loaded": loaded,
        "missing": missing,
        "shape_mismatch": shape_mismatch,
        "checkpoint": str(checkpoint_path),
        "include_backbone": bool(include_backbone),
    }


def _build_trainer(cfg: Dict[str, Any], seed: int, eval_episodes: int, deterministic_eval: bool) -> tuple[TrainerSyncOnPolicy, Any, Any]:
    env = MultiUAV2DEnv(**cfg["env"])
    obs, _ = env.reset(seed=seed)
    agent_id = sorted(obs.keys())[0]
    high_obs = obs[agent_id]["high"]
    low_obs = obs[agent_id]["low"]
    obs_dim_high = int(high_obs.self_state.shape[0] + high_obs.goal_relative.shape[0] + high_obs.neighbor_summary.shape[0])
    obs_dim_low = int(low_obs.flat.shape[0])

    skills = build_default_skill_library(max_duration=int(cfg["skills"]["default_max_duration"]))
    n_skills = len(skills)
    names = list(cfg["skills"].get("names", []))
    if names and len(names) != n_skills:
        raise ValueError(f"skills.names has {len(names)} entries but skill library has {n_skills}")

    high_policy = HighLevelPolicy(
        obs_dim=obs_dim_high,
        n_skills=n_skills,
        hidden_dim=int(cfg["model"]["high_hidden_dim"]),
    )
    low_policy = LowLevelQPPolicy(
        obs_dim=obs_dim_low,
        n_skills=n_skills,
        action_dim=int(cfg["model"]["action_dim"]),
        hidden_dim=int(cfg["model"]["low_hidden_dim"]),
    )

    high_opt = torch.optim.Adam(high_policy.parameters(), lr=3e-4)
    low_opt = torch.optim.Adam(low_policy.parameters(), lr=1e-3)
    mappo = OnPolicyMAPPO(
        policy=high_policy,
        optimizer=high_opt,
        config=MAPPOConfig(**cfg["high_level_mappo"]),
    )

    action_limit = float(cfg["env"]["action_limit"])
    constraint_builder = ConstraintBuilder(
        d_min_agent=float(cfg["safety"]["d_min_agent"]),
        d_safe_obs=float(cfg["safety"]["d_safe_obs"]),
        u_min=[-action_limit, -action_limit],
        u_max=[action_limit, action_limit],
    )
    qp_solver = DifferentiableQPSolver(
        action_dim=int(cfg["model"]["action_dim"]),
        use_stub_if_unavailable=bool(cfg["qp"]["use_stub_if_unavailable"]),
        ecos_max_iters=int(cfg["qp"].get("ecos_max_iters", 500)),
        scs_max_iters=int(cfg["qp"].get("scs_max_iters", 10_000)),
        scs_eps=float(cfg["qp"].get("scs_eps", 1e-4)),
    )
    skill_params = dict(cfg["skills"]["params"])
    skill_params["action_limit"] = action_limit
    skill_params["cbf_u_max"] = action_limit
    runtime = SkillRuntimeManager(
        skills,
        default_ctx=skill_params,
    )
    low_controller = LowLevelSafeController(
        low_policy=low_policy,
        constraint_builder=constraint_builder,
        qp_solver=qp_solver,
        skill_ref_weight=0.7,
        neighbor_perception_radius=float(cfg["env"]["neighbor_radius"]),
        obstacle_perception_range=float(cfg["env"]["lidar_range"]),
    )
    hooks = TrainerHooks(
        rollout_steps=int(cfg["train"]["rollout_steps"]),
        eval_interval=int(cfg["train"]["eval_interval"]),
        gamma_high=float(cfg["train"]["gamma_high"]),
        lam_high=float(cfg["train"]["lam_high"]),
        gamma_low=float(cfg["train"]["gamma_low"]),
        low_ext_reward_coef=float(cfg["train"]["low_ext_reward_coef"]),
        low_update_epochs=int(cfg["train"]["low_update_epochs"]),
        low_max_samples_per_iter=int(cfg["train"]["low_max_samples_per_iter"]),
        low_target_step_scale=float(cfg["train"]["low_target_step_scale"]),
        eval_episodes=int(eval_episodes),
        eval_deterministic=bool(deterministic_eval),
        eval_render=False,
    )
    trainer = TrainerSyncOnPolicy(
        env=env,
        high_policy=high_policy,
        low_policy=low_policy,
        constraint_builder=constraint_builder,
        qp_solver=qp_solver,
        coordinator=SyncCoordinator(
            num_agents=int(cfg["env"]["n_agents"]),
            t_sync_max=int(cfg["synchronization"]["t_sync_max"]),
            mode=str(cfg["synchronization"].get("mode", "sync")),
        ),
        buffer=HierRolloutBuffer(),
        skill_runtime=runtime,
        low_level_controller=low_controller,
        diff_qp_solver=TorchDifferentiableQPSolver(
            action_dim=int(cfg["model"]["action_dim"]),
            ecos_max_iters=int(cfg["qp"].get("ecos_max_iters", 500)),
            scs_max_iters=int(cfg["qp"].get("scs_max_iters", 10_000)),
            scs_eps=float(cfg["qp"].get("scs_eps", 1e-4)),
        ),
        high_level_updater=mappo,
        low_level_optimizer=low_opt,
        hooks=hooks,
    )
    return trainer, high_opt, low_opt


def main() -> None:
    args = _parse_args()

    cfg_path = Path(args.config)
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42) if args.seed < 0 else args.seed)
    if args.total_iterations > 0:
        cfg["train"]["total_iterations"] = int(args.total_iterations)
    if args.eval_interval > 0:
        cfg["train"]["eval_interval"] = int(args.eval_interval)

    _seed_all(seed)

    output_root = Path(args.output_root)
    run_dir = _make_run_dir(output_root=output_root, run_name=args.run_name)
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    trainer, high_opt, low_opt = _build_trainer(
        cfg=cfg,
        seed=seed,
        eval_episodes=max(1, int(args.eval_episodes)),
        deterministic_eval=bool(args.deterministic_eval),
    )
    warm_cfg = dict(cfg.get("warmstart", {}))
    warm_path_arg = str(args.warmstart_low_hfg_from).strip()
    warm_path_cfg = str(warm_cfg.get("low_hfg_checkpoint", "")).strip()
    warm_path = warm_path_arg or warm_path_cfg
    include_backbone = bool(args.warmstart_include_backbone) or bool(warm_cfg.get("include_backbone", False))
    warm_report: Dict[str, Any] = {}
    if warm_path:
        warm_report = _warmstart_low_policy_hfg(
            low_policy=trainer.low_policy,
            checkpoint_path=Path(warm_path),
            include_backbone=include_backbone,
        )
        print(
            "WARMSTART_LOW_HFG "
            f"checkpoint={warm_report['checkpoint']} "
            f"loaded={len(warm_report['loaded'])} "
            f"missing={len(warm_report['missing'])} "
            f"shape_mismatch={len(warm_report['shape_mismatch'])} "
            f"include_backbone={int(bool(warm_report['include_backbone']))}"
        )

    history: list[Dict[str, float]] = []
    last_eval: Dict[str, float] = {}
    eval_success_history: list[float] = []
    total_iterations = int(cfg["train"]["total_iterations"])
    eval_interval = max(1, int(cfg["train"]["eval_interval"]))

    for itr in range(1, total_iterations + 1):
        rollout = trainer.collect_rollout()
        low = trainer.update_low_level()
        high = trainer.update_high_level()
        row = {
            "iteration": float(itr),
            "steps_collected": float(rollout.get("steps_collected", 0.0)),
            "episode_return_mean": float(rollout.get("episode_return_mean", 0.0)),
            "safe_reach_ratio": float(rollout.get("safe_reach_ratio", 0.0)),
            "conv_eval_success_delta_w5": float("nan"),
            "high_samples": float(rollout.get("high_samples", 0.0)),
            "low_samples": float(rollout.get("low_samples", 0.0)),
            "loss_high_total": float(high.get("loss_total", 0.0)),
            "loss_low_mean": float(low.get("loss_mean", 0.0)),
        }
        if itr == 1 or itr % eval_interval == 0 or itr == total_iterations:
            last_eval = trainer.evaluate()
            row.update({k: float(v) for k, v in last_eval.items()})
            eval_success_history.append(float(last_eval.get("eval_success_rate", 0.0)))
            if len(eval_success_history) >= 10:
                prev = float(np.mean(eval_success_history[-10:-5]))
                recent = float(np.mean(eval_success_history[-5:]))
                row["conv_eval_success_delta_w5"] = abs(recent - prev)
            print(
                f"[iter {itr}/{total_iterations}] "
                f"ret={row['episode_return_mean']:.4f} "
                f"safe={row['safe_reach_ratio']:.4f} "
                f"succ={float(last_eval.get('eval_success_rate', 0.0)):.4f} "
                f"coll={float(last_eval.get('eval_collision_rate', 0.0)):.4f} "
                f"conv={row['conv_eval_success_delta_w5']:.4f}"
            )
        history.append(row)

    trainer.hooks.eval_episodes = max(1, int(args.final_eval_episodes))
    trainer.hooks.eval_render = True
    trainer.hooks.eval_render_gif = bool(args.final_render_gif) or not bool(args.final_render_png)
    trainer.hooks.eval_render_dir = str(run_dir / "eval_media")
    final_eval = trainer.evaluate()

    torch.save(
        {
            "seed": seed,
            "config": cfg,
            "high_policy": trainer.high_policy.state_dict(),
            "low_policy": trainer.low_policy.state_dict(),
            "high_optimizer": high_opt.state_dict(),
            "low_optimizer": low_opt.state_dict(),
        },
        run_dir / "checkpoints" / "last.pt",
    )
    _write_history_csv(run_dir / "train_history.csv", history)
    summary = {
        "seed": seed,
        "total_iterations": total_iterations,
        "final_eval": final_eval,
        "last_eval_during_train": last_eval,
        "run_dir": str(run_dir),
        "warmstart_low_hfg": warm_report,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "LATEST_RUN").write_text(str(run_dir), encoding="utf-8")

    media = sorted((run_dir / "eval_media").glob("*.gif")) + sorted((run_dir / "eval_media").glob("*.png"))
    print(f"RUN_DIR={run_dir}")
    if media:
        print("FINAL_MEDIA:")
        for path in media:
            print(str(path))
    else:
        print("FINAL_MEDIA: none")


if __name__ == "__main__":
    main()
