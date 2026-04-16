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
from hmarl_cbf.baselines import DistributedCBFBaselineConfig, DistributedCBFBaselineController
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
from hmarl_cbf.scenarios import build_fixed_scene
from hmarl_cbf.skills import SkillRuntimeManager, build_default_skill_library
from hmarl_cbf.train import TrainerHooks, TrainerSyncOnPolicy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HMARL-CBF and export final evaluation video.")
    parser.add_argument("--config", type=str, default="configs/hmarl_cbf/default_async_onpolicy_gcbfplus.yaml")
    parser.add_argument("--output-root", type=str, default="artifacts/hmarl_cbf_paper_async")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--total-iterations", type=int, default=-1)
    parser.add_argument("--eval-interval", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--low-update-mode", type=str, default="", help="Override low-level update mode.")
    parser.add_argument("--low-ppo-epochs", type=int, default=-1, help="Override low-level PPO epochs.")
    parser.add_argument("--low-policy-action-std", type=float, default=-1.0, help="Override low-level action std.")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Periodic evaluation episodes during training.")
    parser.add_argument(
        "--video-interval",
        type=int,
        default=0,
        help="If >0, render periodic evaluation GIFs every N iterations during training.",
    )
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
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Resume full training state from an existing checkpoint.",
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


def _infer_checkpoint_iteration(checkpoint_path: Path, ckpt: Dict[str, Any]) -> int:
    if "last_iteration" in ckpt:
        try:
            return int(ckpt["last_iteration"])
        except Exception:
            pass
    run_dir = checkpoint_path.parents[1] if checkpoint_path.name == "last.pt" else checkpoint_path.parent
    history_path = run_dir / "train_history.csv"
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                return int(float(rows[-1].get("iteration", "0") or 0))
        except Exception:
            pass
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if "total_iterations" in payload:
                return int(payload["total_iterations"])
        except Exception:
            pass
    return 0


def _resume_full_training_state(
    trainer: TrainerSyncOnPolicy,
    high_opt: Any,
    low_opt: Any,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"invalid resume checkpoint: {checkpoint_path}")
    if "high_policy" not in ckpt or "low_policy" not in ckpt:
        raise ValueError("resume checkpoint must contain high_policy and low_policy")
    trainer.high_policy.load_state_dict(ckpt["high_policy"], strict=True)
    trainer.low_policy.load_state_dict(ckpt["low_policy"], strict=True)
    loaded_high_opt = False
    loaded_low_opt = False
    if "high_optimizer" in ckpt:
        high_opt.load_state_dict(ckpt["high_optimizer"])
        loaded_high_opt = True
    if "low_optimizer" in ckpt:
        low_opt.load_state_dict(ckpt["low_optimizer"])
        loaded_low_opt = True
    return {
        "checkpoint": str(checkpoint_path),
        "resume_iteration": int(_infer_checkpoint_iteration(checkpoint_path, ckpt)),
        "loaded_high_optimizer": bool(loaded_high_opt),
        "loaded_low_optimizer": bool(loaded_low_opt),
    }


def _set_high_entropy_coef(trainer: TrainerSyncOnPolicy, train_cfg: Dict[str, Any], itr: int, total_iterations: int) -> float:
    if trainer.high_level_updater is None:
        return float(train_cfg.get("high_entropy_coef_end", train_cfg.get("high_entropy_coef_start", 0.0)))
    default_coef = float(trainer.high_level_updater.config.entropy_coef)
    start = float(train_cfg.get("high_entropy_coef_start", default_coef))
    end = float(train_cfg.get("high_entropy_coef_end", start))
    decay_iters = max(1, int(train_cfg.get("high_entropy_decay_iters", total_iterations)))
    if decay_iters <= 1:
        coef = end
    else:
        alpha = min(1.0, max(0.0, float(itr - 1) / float(decay_iters - 1)))
        coef = start + (end - start) * alpha
    trainer.high_level_updater.config.entropy_coef = float(coef)
    return float(coef)


def _set_low_entropy_coef(trainer: TrainerSyncOnPolicy, train_cfg: Dict[str, Any], itr: int, total_iterations: int) -> float:
    default_coef = float(trainer.hooks.low_ppo_entropy_coef)
    start = float(train_cfg.get("low_ppo_entropy_coef_start", default_coef))
    end = float(train_cfg.get("low_ppo_entropy_coef_end", train_cfg.get("low_ppo_entropy_coef", start)))
    decay_iters = max(1, int(train_cfg.get("low_ppo_entropy_decay_iters", total_iterations)))
    if decay_iters <= 1:
        coef = end
    else:
        alpha = min(1.0, max(0.0, float(itr - 1) / float(decay_iters - 1)))
        coef = start + (end - start) * alpha
    trainer.hooks.low_ppo_entropy_coef = float(coef)
    return float(coef)


def _set_low_update_mode_schedule(trainer: TrainerSyncOnPolicy, train_cfg: Dict[str, Any], itr: int) -> str:
    main_mode = str(train_cfg.get("low_update_mode_main", train_cfg.get("low_update_mode", "onpolicy_ppo"))).strip()
    pretrain_mode = str(train_cfg.get("low_update_mode_pretrain", "reference_regression")).strip()
    pretrain_iters = max(0, int(train_cfg.get("low_reference_pretrain_iters", 0)))
    if pretrain_iters > 0 and itr <= pretrain_iters:
        trainer.hooks.low_update_mode = pretrain_mode
    else:
        trainer.hooks.low_update_mode = main_mode
    return str(trainer.hooks.low_update_mode)


class _FixedSceneMixer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        mix_cfg = dict(cfg.get("scenario_mix", {}))
        self.enabled = bool(mix_cfg.get("enabled", False))
        self.general_random_prob = float(mix_cfg.get("general_random_prob", 1.0))
        self.entries: list[tuple[str, float]] = []
        raw_entries = list(mix_cfg.get("scenarios", []))
        for item in raw_entries:
            if isinstance(item, str):
                self.entries.append((str(item), 1.0))
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                prob = float(item.get("prob", 0.0))
                if name:
                    self.entries.append((name, prob))
        total = self.general_random_prob + sum(prob for _, prob in self.entries)
        self.total_weight = float(total)

    def __call__(self, env: MultiUAV2DEnv) -> tuple[str, Dict[str, Any] | None] | None:
        if not self.enabled or self.total_weight <= 0.0:
            return None
        draw = random.random() * self.total_weight
        cursor = self.general_random_prob
        if draw < cursor:
            return "random", None
        for name, prob in self.entries:
            cursor += prob
            if draw <= cursor:
                states, obstacles = build_fixed_scene(
                    name=name,
                    world_size=float(env.world_size),
                    agent_radius=float(env.agent_radius),
                )
                if len(states) != int(env.n_agents):
                    raise ValueError(
                        f"scenario_mix scene {name} has {len(states)} agents but env expects {env.n_agents}"
                    )
                return str(name), {"states": states, "obstacles": obstacles}
        return "random", None


def _set_training_curriculum_stage(trainer: TrainerSyncOnPolicy, cfg: Dict[str, Any], itr: int) -> str:
    curriculum_cfg = dict(cfg.get("curriculum", {}))
    if not bool(curriculum_cfg.get("enabled", False)):
        return "full"

    single_agent_no_obstacle_iters = int(curriculum_cfg.get("single_agent_no_obstacle_iters", 0))
    if single_agent_no_obstacle_iters <= 0 or itr > single_agent_no_obstacle_iters:
        use_single_agent_stage = False
    else:
        prob_start = float(curriculum_cfg.get("single_agent_no_obstacle_prob_start", 0.85))
        prob_end = float(curriculum_cfg.get("single_agent_no_obstacle_prob_end", 0.15))
        if single_agent_no_obstacle_iters <= 1:
            mix_prob = prob_end
        else:
            alpha = min(1.0, max(0.0, float(itr - 1) / float(single_agent_no_obstacle_iters - 1)))
            mix_prob = prob_start + (prob_end - prob_start) * alpha
        mix_prob = float(np.clip(mix_prob, 0.0, 1.0))
        use_single_agent_stage = bool(np.random.rand() < mix_prob)

    if not hasattr(trainer, "_full_env"):
        trainer._full_env = trainer.env  # type: ignore[attr-defined]
        trainer._full_coordinator = trainer.coordinator  # type: ignore[attr-defined]
        trainer._curriculum_cache = {}  # type: ignore[attr-defined]

    if not use_single_agent_stage:
        trainer.env = trainer._full_env  # type: ignore[attr-defined]
        trainer.coordinator = trainer._full_coordinator  # type: ignore[attr-defined]
        return "full"

    cache = trainer._curriculum_cache  # type: ignore[attr-defined]
    key = ("single_agent_no_obstacle", 1, 0)
    if key not in cache:
        env_cfg = dict(cfg["env"])
        env_cfg["n_agents"] = 1
        env_cfg["n_obstacles"] = 0
        env_stage = MultiUAV2DEnv(**env_cfg)
        coord_stage = SyncCoordinator(
            num_agents=1,
            t_sync_max=int(cfg["synchronization"]["t_sync_max"]),
            mode=str(cfg["synchronization"].get("mode", "sync")),
        )
        cache[key] = (env_stage, coord_stage)
    trainer.env, trainer.coordinator = cache[key]
    return "single_agent_no_obstacle"


def _restore_full_training_env(trainer: TrainerSyncOnPolicy) -> None:
    if hasattr(trainer, "_full_env"):
        trainer.env = trainer._full_env  # type: ignore[attr-defined]
    if hasattr(trainer, "_full_coordinator"):
        trainer.coordinator = trainer._full_coordinator  # type: ignore[attr-defined]


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
        "phi_mu_head.",
        "phi_log_std_head.",
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
        h_diag_min=float(cfg.get("low_level_qp", {}).get("h_diag_min", 1e-2)),
        h_diag_max=float(cfg.get("low_level_qp", {}).get("h_diag_max", 50.0)),
        h_offdiag_abs_max=float(cfg.get("low_level_qp", {}).get("h_offdiag_abs_max", 5.0)),
        f_abs_max=float(cfg.get("low_level_qp", {}).get("f_abs_max", 20.0)),
        phi_log_std_min=float(cfg.get("low_level_qp", {}).get("phi_log_std_min", -5.0)),
        phi_log_std_max=float(cfg.get("low_level_qp", {}).get("phi_log_std_max", 1.0)),
        w_clf=float(cfg.get("low_level_qp", {}).get("w_clf", 10.0)),
        w_cbf=float(cfg.get("low_level_qp", {}).get("w_cbf", 100.0)),
        cbf_slack_max=float(cfg.get("low_level_qp", {}).get("cbf_slack_max", 1.0)),
        cbf_k0=float(cfg.get("low_level_qp", {}).get("cbf_k0", 1.0)),
        cbf_k1=float(cfg.get("low_level_qp", {}).get("cbf_k1", 1.0)),
        clf_k=float(cfg.get("low_level_qp", {}).get("clf_k", 1.0)),
        hocbf_gamma_h=float(cfg.get("low_level_qp", {}).get("hocbf_gamma_h", 1.0)),
        hocbf_gamma_hdot=float(cfg.get("low_level_qp", {}).get("hocbf_gamma_hdot", 1.0)),
        f_residual_reference_enabled=bool(cfg.get("low_level_qp", {}).get("f_residual_reference_enabled", True)),
        f_ref_speed=float(cfg.get("low_level_qp", {}).get("f_ref_speed", cfg["skills"]["params"].get("ref_speed", 1.2))),
        f_ref_kp=float(cfg.get("low_level_qp", {}).get("f_ref_kp", 1.2)),
        f_ref_slow_radius=float(cfg.get("low_level_qp", {}).get("f_ref_slow_radius", cfg["skills"]["params"].get("slow_radius", 1.5))),
        f_ref_goal_stop_min_speed=float(cfg.get("low_level_qp", {}).get("f_ref_goal_stop_min_speed", cfg["skills"]["params"].get("goal_stop_min_speed", 0.0))),
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
    teacher_baseline_cfg = DistributedCBFBaselineConfig.from_mapping(
        {
            "cbf_mode": cfg["skills"]["params"].get("cbf_mode", "distributed_gcbfplus"),
            "cbf_share_agent": cfg["skills"]["params"].get("cbf_share_agent", 0.5),
            "cbf_share_obs": cfg["skills"]["params"].get("cbf_share_obs", 1.0),
            "cbf_u_max": action_limit,
            "cbf_k0": cfg.get("low_level_qp", {}).get("cbf_k0", 1.0),
            "cbf_k1": cfg.get("low_level_qp", {}).get("cbf_k1", 1.0),
            "hocbf_gamma_h": cfg.get("low_level_qp", {}).get("hocbf_gamma_h", 1.0),
            "hocbf_gamma_hdot": cfg.get("low_level_qp", {}).get("hocbf_gamma_hdot", 1.0),
            "clf_k": cfg.get("low_level_qp", {}).get("clf_k", 1.0),
            "H_diag": [1.0, 1.0],
            "w_clf": cfg.get("low_level_qp", {}).get("w_clf", 10.0),
            "w_cbf": cfg.get("low_level_qp", {}).get("w_cbf", 100.0),
            "cbf_slack_max": cfg.get("low_level_qp", {}).get("cbf_slack_max", 1.0),
            "ref_speed": cfg["skills"]["params"].get("ref_speed", 1.2),
            "speed_kp": cfg.get("teacher_baseline", {}).get("speed_kp", 1.2),
            "slow_radius": cfg["skills"]["params"].get("slow_radius", 1.5),
            "goal_stop_min_speed": cfg["skills"]["params"].get("goal_stop_min_speed", 0.0),
            "neighbor_radius": cfg["env"].get("neighbor_radius", 2.0),
            "obstacle_range": cfg["env"].get("lidar_range", 3.0),
            "boundary_cbf": cfg.get("safety", {}).get("boundary_cbf", True),
            "boundary_margin": cfg.get("safety", {}).get("boundary_margin", cfg["env"].get("agent_radius", 0.2)),
            "world_size": cfg["env"].get("world_size", 10.0),
            "rect_base_margin_extra": cfg["env"].get("rect_base_margin_extra", 0.0),
            "rect_corner_margin_enabled": cfg["env"].get("rect_corner_margin_enabled", False),
            "rect_corner_margin_max": cfg["env"].get("rect_corner_margin_max", 0.0),
            "rect_corner_proximity_distance": cfg["env"].get("rect_corner_proximity_distance", 0.4),
            "rect_corner_speed_min": cfg["env"].get("rect_corner_speed_min", 0.05),
            "rect_corner_alignment_power": cfg["env"].get("rect_corner_alignment_power", 1.0),
            "use_input_bounds": True,
        }
    )
    teacher_baseline_controller = DistributedCBFBaselineController(
        constraint_builder=constraint_builder,
        qp_solver=qp_solver,
        config=teacher_baseline_cfg,
    )
    skill_params = dict(cfg["skills"]["params"])
    skill_params["action_limit"] = action_limit
    skill_params["cbf_u_max"] = action_limit
    skill_params["dt"] = float(cfg["env"]["dt"])
    skill_params["world_size"] = float(cfg["env"]["world_size"])
    skill_params["boundary_cbf"] = bool(cfg.get("safety", {}).get("boundary_cbf", True))
    skill_params["boundary_margin"] = float(
        cfg.get("safety", {}).get("boundary_margin", cfg["env"].get("agent_radius", 0.2))
    )
    skill_params["rect_base_margin_extra"] = float(cfg["env"].get("rect_base_margin_extra", 0.0))
    skill_params["rect_corner_margin_enabled"] = bool(cfg["env"].get("rect_corner_margin_enabled", False))
    skill_params["rect_corner_margin_max"] = float(cfg["env"].get("rect_corner_margin_max", 0.0))
    skill_params["rect_corner_proximity_distance"] = float(cfg["env"].get("rect_corner_proximity_distance", 0.4))
    skill_params["rect_corner_speed_min"] = float(cfg["env"].get("rect_corner_speed_min", 0.05))
    skill_params["rect_corner_alignment_power"] = float(cfg["env"].get("rect_corner_alignment_power", 1.0))
    runtime = SkillRuntimeManager(
        skills,
        default_ctx=skill_params,
    )
    low_controller = LowLevelSafeController(
        low_policy=low_policy,
        constraint_builder=constraint_builder,
        qp_solver=qp_solver,
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
        low_reward_mix_eta=float(cfg["train"].get("low_reward_mix_eta", 0.0)),
        low_reward_mix_divide_by_n_agents=bool(cfg["train"].get("low_reward_mix_divide_by_n_agents", True)),
        low_safety_margin_coef=float(cfg["train"].get("low_safety_margin_coef", 0.0)),
        low_safety_margin_h_agent=float(cfg["train"].get("low_safety_margin_h_agent", 0.0)),
        low_safety_margin_h_obstacle=float(cfg["train"].get("low_safety_margin_h_obstacle", 0.0)),
        high_option_progress_coef=float(cfg["train"].get("high_option_progress_coef", 0.0)),
        high_option_boundary_recovery_coef=float(cfg["train"].get("high_option_boundary_recovery_coef", 0.0)),
        high_option_boundary_threshold=float(cfg["train"].get("high_option_boundary_threshold", 0.0)),
        high_option_trap_relief_coef=float(cfg["train"].get("high_option_trap_relief_coef", 0.0)),
        high_option_trap_enter_coef=float(cfg["train"].get("high_option_trap_enter_coef", 0.0)),
        high_option_stuck_penalty_coef=float(cfg["train"].get("high_option_stuck_penalty_coef", 0.0)),
        high_option_stuck_blocked_threshold=float(cfg["train"].get("high_option_stuck_blocked_threshold", 0.5)),
        high_option_stuck_progress_threshold=float(cfg["train"].get("high_option_stuck_progress_threshold", 0.1)),
        high_option_stuck_speed_threshold=float(cfg["train"].get("high_option_stuck_speed_threshold", 0.2)),
        high_trap_blocked_lookahead=float(cfg["train"].get("high_trap_blocked_lookahead", 4.0)),
        high_trap_blocked_lateral_window=float(cfg["train"].get("high_trap_blocked_lateral_window", 3.0)),
        high_trap_blocked_extra_margin=float(cfg["train"].get("high_trap_blocked_extra_margin", 0.1)),
        low_update_epochs=int(cfg["train"]["low_update_epochs"]),
        low_max_samples_per_iter=int(cfg["train"]["low_max_samples_per_iter"]),
        low_target_step_scale=float(cfg["train"]["low_target_step_scale"]),
        low_update_mode=str(cfg["train"].get("low_update_mode", "deterministic_diff")),
        low_ppo_epochs=int(cfg["train"].get("low_ppo_epochs", cfg["train"].get("low_update_epochs", 2))),
        low_ppo_clip_ratio=float(cfg["train"].get("low_ppo_clip_ratio", 0.2)),
        low_ppo_value_coef=float(cfg["train"].get("low_ppo_value_coef", 0.5)),
        low_ppo_entropy_coef=float(cfg["train"].get("low_ppo_entropy_coef", 0.0)),
        low_ppo_max_grad_norm=float(cfg["train"].get("low_ppo_max_grad_norm", 0.5)),
        low_policy_action_std=float(cfg["train"].get("low_policy_action_std", 0.2)),
        low_normalize_advantages=bool(cfg["train"].get("low_normalize_advantages", True)),
        low_deterministic_value_coef=float(cfg["train"].get("low_deterministic_value_coef", cfg["train"].get("low_ppo_value_coef", 0.5))),
        low_deterministic_slack_coef=float(cfg["train"].get("low_deterministic_slack_coef", 0.02)),
        low_deterministic_cbf_slack_coef=float(cfg["train"].get("low_deterministic_cbf_slack_coef", 0.05)),
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
        teacher_baseline_controller=teacher_baseline_controller,
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
    if str(args.low_update_mode).strip():
        cfg["train"]["low_update_mode"] = str(args.low_update_mode).strip()
    if args.low_ppo_epochs > 0:
        cfg["train"]["low_ppo_epochs"] = int(args.low_ppo_epochs)
    if args.low_policy_action_std > 0:
        cfg["train"]["low_policy_action_std"] = float(args.low_policy_action_std)

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
    trainer.set_training_scene_sampler(_FixedSceneMixer(cfg))

    resume_report: Dict[str, Any] = {}
    resume_path = str(args.resume_from).strip()
    if resume_path:
        resume_report = _resume_full_training_state(
            trainer=trainer,
            high_opt=high_opt,
            low_opt=low_opt,
            checkpoint_path=Path(resume_path),
        )
        print(
            "RESUME_FULL "
            f"checkpoint={resume_report['checkpoint']} "
            f"resume_iteration={resume_report['resume_iteration']} "
            f"high_opt={int(bool(resume_report['loaded_high_optimizer']))} "
            f"low_opt={int(bool(resume_report['loaded_low_optimizer']))}"
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
    resume_iteration = int(resume_report.get("resume_iteration", 0))
    display_total_iterations = int(resume_iteration + total_iterations)
    eval_interval = max(1, int(cfg["train"]["eval_interval"]))
    video_interval = max(0, int(args.video_interval))

    for itr_local in range(1, total_iterations + 1):
        itr = int(resume_iteration + itr_local)
        curriculum_stage = _set_training_curriculum_stage(trainer, cfg, itr)
        high_entropy_coef = _set_high_entropy_coef(trainer, cfg["train"], itr, display_total_iterations)
        low_entropy_coef = _set_low_entropy_coef(trainer, cfg["train"], itr, display_total_iterations)
        low_update_mode_stage = _set_low_update_mode_schedule(trainer, cfg["train"], itr)
        rollout = trainer.collect_rollout()
        _restore_full_training_env(trainer)
        low = trainer.update_low_level()
        high = trainer.update_high_level()
        row = {
            "iteration": float(itr),
            "curriculum_single_agent_stage": float(1.0 if curriculum_stage == "single_agent_no_obstacle" else 0.0),
            "mixed_fixed_scene_stage": float(0.0 if str(getattr(trainer, "last_rollout_scene_name", "random")) == "random" else 1.0),
            "steps_collected": float(rollout.get("steps_collected", 0.0)),
            "episode_return_mean": float(rollout.get("episode_return_mean", 0.0)),
            "safe_reach_ratio": float(rollout.get("safe_reach_ratio", 0.0)),
            "skill_entropy_norm": float(rollout.get("skill_entropy_norm", 0.0)),
            "top1_skill_ratio": float(rollout.get("top1_skill_ratio", 0.0)),
            "high_div_bonus_mean": float(rollout.get("high_div_bonus_mean", 0.0)),
            "conv_eval_success_delta_w5": float("nan"),
            "high_entropy_coef": float(high_entropy_coef),
            "low_entropy_coef": float(low_entropy_coef),
            "low_reference_pretrain_stage": float(
                1.0 if str(low_update_mode_stage).strip().lower() in {"reference_regression", "reference_pretrain"} else 0.0
            ),
            "high_samples": float(rollout.get("high_samples", 0.0)),
            "low_samples": float(rollout.get("low_samples", 0.0)),
            "loss_high_total": float(high.get("loss_total", 0.0)),
            "loss_low_mean": float(low.get("loss_mean", 0.0)),
            "loss_low_actor": float(low.get("loss_actor", 0.0)),
            "loss_low_value": float(low.get("loss_value", 0.0)),
            "low_entropy": float(low.get("entropy", 0.0)),
            "low_f_mean_x": float(low.get("low_f_mean_x", 0.0)),
            "low_f_mean_y": float(low.get("low_f_mean_y", 0.0)),
            "low_h_eig_min": float(low.get("low_h_eig_min", 0.0)),
            "low_h_eig_max": float(low.get("low_h_eig_max", 0.0)),
        }
        should_eval = bool(itr_local == 1 or itr_local % eval_interval == 0 or itr_local == total_iterations)
        should_video = bool(video_interval > 0 and (itr_local % video_interval == 0))

        if should_eval:
            prev_render = bool(trainer.hooks.eval_render)
            prev_render_gif = bool(trainer.hooks.eval_render_gif)
            prev_render_dir = str(trainer.hooks.eval_render_dir)
            if should_video:
                trainer.hooks.eval_render = True
                trainer.hooks.eval_render_gif = True
                trainer.hooks.eval_render_dir = str(run_dir / "eval_media" / f"iter_{itr:04d}")

            last_eval = trainer.evaluate()

            trainer.hooks.eval_render = prev_render
            trainer.hooks.eval_render_gif = prev_render_gif
            trainer.hooks.eval_render_dir = prev_render_dir

            row.update({k: float(v) for k, v in last_eval.items()})
            eval_success_history.append(float(last_eval.get("eval_success_rate", 0.0)))
            if len(eval_success_history) >= 10:
                prev = float(np.mean(eval_success_history[-10:-5]))
                recent = float(np.mean(eval_success_history[-5:]))
                row["conv_eval_success_delta_w5"] = abs(recent - prev)
            print(
                f"[iter {itr}/{display_total_iterations}] "
                f"ret={row['episode_return_mean']:.4f} "
                f"safe={row['safe_reach_ratio']:.4f} "
                f"skillH={row['skill_entropy_norm']:.4f} "
                f"low_actor={row['loss_low_actor']:.4f} "
                f"low_entropy={row['low_entropy']:.4f} "
                f"loss_slack={float(low.get('loss_slack', 0.0)):.4f} "
                f"Fx={row['low_f_mean_x']:.3f} "
                f"Fy={row['low_f_mean_y']:.3f} "
                f"Hmin={row['low_h_eig_min']:.3f} "
                f"Hmax={row['low_h_eig_max']:.3f}"
            )
            if should_video:
                print(f"[iter {itr}/{display_total_iterations}] periodic_media_dir={run_dir / 'eval_media' / f'iter_{itr:04d}'}")
        elif should_video:
            prev_render = bool(trainer.hooks.eval_render)
            prev_render_gif = bool(trainer.hooks.eval_render_gif)
            prev_render_dir = str(trainer.hooks.eval_render_dir)
            trainer.hooks.eval_render = True
            trainer.hooks.eval_render_gif = True
            trainer.hooks.eval_render_dir = str(run_dir / "eval_media" / f"iter_{itr:04d}")
            _ = trainer.evaluate()
            trainer.hooks.eval_render = prev_render
            trainer.hooks.eval_render_gif = prev_render_gif
            trainer.hooks.eval_render_dir = prev_render_dir
            print(f"[iter {itr}/{display_total_iterations}] periodic_media_dir={run_dir / 'eval_media' / f'iter_{itr:04d}'}")
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
            "last_iteration": int(display_total_iterations),
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
        "resume_full": resume_report,
        "resume_iteration": int(resume_iteration),
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
