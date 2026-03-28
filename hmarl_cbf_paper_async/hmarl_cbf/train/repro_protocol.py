from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, List, Mapping


@dataclass(slots=True)
class ReproExperiment:
    name: str
    description: str = ""
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReproSuiteConfig:
    name: str
    seeds: List[int]
    total_iterations: int
    eval_interval: int
    output_dir: str
    experiments: List[ReproExperiment]


@dataclass(slots=True)
class ReproRunRecord:
    suite_name: str
    experiment: str
    seed: int
    total_iterations: int
    final_success_rate: float
    final_collision_rate: float
    final_reach_rate: float
    final_qp_feasible_rate: float
    final_avg_episode_return: float
    final_avg_steps: float
    final_loss_high: float
    final_loss_low: float


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def run_single_seed(
    trainer: Any,
    total_iterations: int,
    eval_interval: int,
) -> Dict[str, Any]:
    history: List[Dict[str, float]] = []
    last_eval: Dict[str, float] = {}
    last_high: Dict[str, float] = {}
    last_low: Dict[str, float] = {}

    for itr in range(int(total_iterations)):
        rollout = trainer.collect_rollout()
        low = trainer.update_low_level()
        high = trainer.update_high_level()
        row = {
            "iteration": float(itr + 1),
            "steps_collected": _safe_float(rollout.get("steps_collected", 0.0)),
            "episode_return_mean": _safe_float(rollout.get("episode_return_mean", 0.0)),
            "safe_reach_ratio": _safe_float(rollout.get("safe_reach_ratio", 0.0)),
            "high_samples": _safe_float(rollout.get("high_samples", 0.0)),
            "low_samples": _safe_float(rollout.get("low_samples", 0.0)),
            "loss_high_total": _safe_float(high.get("loss_total", 0.0)),
            "loss_low_mean": _safe_float(low.get("loss_mean", 0.0)),
        }
        if (itr + 1) % max(1, int(eval_interval)) == 0:
            last_eval = trainer.evaluate()
            for k, v in last_eval.items():
                row[f"eval_{k}"] = _safe_float(v)
        history.append(row)
        last_high = high
        last_low = low

    if not last_eval:
        last_eval = trainer.evaluate()

    return {
        "history": history,
        "final_eval": last_eval,
        "last_high": last_high,
        "last_low": last_low,
    }


def write_history_csv(history: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    rows = list(history)
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(records: Iterable[ReproRunRecord], path: str | Path) -> None:
    rows = [asdict(r) for r in records]
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_repro_suite(
    config: ReproSuiteConfig,
    trainer_factory: Callable[[int, ReproExperiment], Any],
) -> Dict[str, Any]:
    out_root = _ensure_dir(config.output_dir)
    records: List[ReproRunRecord] = []

    for exp in config.experiments:
        for seed in config.seeds:
            trainer = trainer_factory(seed, exp)
            run = run_single_seed(
                trainer=trainer,
                total_iterations=config.total_iterations,
                eval_interval=config.eval_interval,
            )

            run_dir = out_root / exp.name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_history_csv(run["history"], run_dir / "train_history.csv")
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "final_eval": run["final_eval"],
                        "last_high": run["last_high"],
                        "last_low": run["last_low"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            final_eval = run["final_eval"]
            last_high = run["last_high"]
            last_low = run["last_low"]
            records.append(
                ReproRunRecord(
                    suite_name=config.name,
                    experiment=exp.name,
                    seed=int(seed),
                    total_iterations=int(config.total_iterations),
                    final_success_rate=_safe_float(final_eval.get("eval_success_rate", final_eval.get("success_rate", 0.0))),
                    final_collision_rate=_safe_float(final_eval.get("eval_collision_rate", final_eval.get("collision_rate", 0.0))),
                    final_reach_rate=_safe_float(final_eval.get("eval_reach_rate", final_eval.get("reach_rate", 0.0))),
                    final_qp_feasible_rate=_safe_float(final_eval.get("eval_qp_feasible_rate", final_eval.get("qp_feasible_rate", 0.0))),
                    final_avg_episode_return=_safe_float(
                        final_eval.get("eval_avg_episode_return", final_eval.get("avg_episode_return", 0.0))
                    ),
                    final_avg_steps=_safe_float(final_eval.get("eval_avg_steps", final_eval.get("avg_steps", 0.0))),
                    final_loss_high=_safe_float(last_high.get("loss_total", 0.0)),
                    final_loss_low=_safe_float(last_low.get("loss_mean", 0.0)),
                )
            )

    summary_csv = out_root / "repro_summary.csv"
    write_summary_csv(records, summary_csv)

    grouped: Dict[str, List[ReproRunRecord]] = {}
    for record in records:
        grouped.setdefault(record.experiment, []).append(record)

    aggregate = {}
    for exp_name, rows in grouped.items():
        succ = [r.final_success_rate for r in rows]
        col = [r.final_collision_rate for r in rows]
        ret = [r.final_avg_episode_return for r in rows]
        aggregate[exp_name] = {
            "n_seeds": len(rows),
            "success_rate_mean": mean(succ),
            "success_rate_std": pstdev(succ) if len(succ) > 1 else 0.0,
            "collision_rate_mean": mean(col),
            "collision_rate_std": pstdev(col) if len(col) > 1 else 0.0,
            "avg_episode_return_mean": mean(ret),
            "avg_episode_return_std": pstdev(ret) if len(ret) > 1 else 0.0,
        }

    summary_json = out_root / "repro_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "suite_name": config.name,
                "total_runs": len(records),
                "experiments": aggregate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "n_runs": float(len(records)),
    }
