from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime entrypoint
    raise RuntimeError("PyTorch is required to run training") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime entrypoint
    raise RuntimeError("PyYAML is required to load training config") from exc

from hmarl_cbf.train.run_sync_onpolicy import (
    _FixedSceneMixer,
    _build_trainer,
    _make_run_dir,
    _resume_full_training_state,
    _restore_full_training_env,
    _seed_all,
    _set_high_entropy_coef,
    _set_low_entropy_coef,
    _set_low_update_mode_schedule,
    _set_training_curriculum_stage,
    _write_history_csv,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the paper-aligned async HMARL-CBF subset with stochastic phi low-level PPO."
    )
    parser.add_argument("--config", type=str, default="configs/hmarl_cbf/default_async_onpolicy_gcbfplus.yaml")
    parser.add_argument("--output-root", type=str, default="artifacts/hmarl_cbf_paper_async")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--total-iterations", type=int, default=-1)
    parser.add_argument("--eval-interval", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--video-interval", type=int, default=0)
    parser.add_argument("--final-eval-episodes", type=int, default=3)
    parser.add_argument("--final-render-gif", action="store_true")
    parser.add_argument("--final-render-png", action="store_true")
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--resume-from", type=str, default="", help="Resume full training state from an existing checkpoint.")
    return parser.parse_args()


def _force_paper_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg)
    cfg["synchronization"] = dict(cfg["synchronization"])
    cfg["synchronization"]["mode"] = "async"
    cfg["train"] = dict(cfg["train"])
    cfg["train"]["low_update_mode"] = "onpolicy_ppo"
    return cfg


def main() -> None:
    args = _parse_args()

    cfg_path = Path(args.config)
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = _force_paper_defaults(cfg)

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
                prev = float(sum(eval_success_history[-10:-5]) / 5.0)
                recent = float(sum(eval_success_history[-5:]) / 5.0)
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
        "paper_subset": True,
        "synchronization_mode": str(cfg["synchronization"].get("mode", "async")),
        "low_update_mode": str(cfg["train"].get("low_update_mode", "onpolicy_ppo")),
        "resume_full": resume_report,
        "resume_iteration": int(resume_iteration),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "LATEST_RUN").write_text(str(run_dir), encoding="utf-8")

    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
