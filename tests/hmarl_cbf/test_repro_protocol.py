from pathlib import Path

from hmarl_cbf.train.repro_protocol import ReproExperiment, ReproSuiteConfig, run_repro_suite


class DummyTrainer:
    def __init__(self, seed: int, scale: float) -> None:
        self.seed = seed
        self.scale = scale
        self.iteration = 0

    def collect_rollout(self):
        self.iteration += 1
        return {"steps_collected": 5.0, "episode_return_mean": self.scale + 0.1 * self.iteration, "high_samples": 4.0, "low_samples": 8.0}

    def update_low_level(self):
        return {"loss_mean": 0.2 * self.scale}

    def update_high_level(self):
        return {"loss_total": 0.1 * self.scale}

    def evaluate(self):
        s = 0.5 + 0.1 * self.scale
        c = 0.2 - 0.05 * self.scale
        return {
            "eval_success_rate": s,
            "eval_collision_rate": c,
            "eval_reach_rate": s,
            "eval_qp_feasible_rate": 1.0,
            "eval_avg_episode_return": 1.0 + self.scale,
            "eval_avg_steps": 10.0,
        }


def test_run_repro_suite_outputs_files() -> None:
    out_dir = Path("tests/.tmp_repro_protocol")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ReproSuiteConfig(
        name="dummy_suite",
        seeds=[0, 1],
        total_iterations=3,
        eval_interval=2,
        output_dir=str(out_dir / "repro"),
        experiments=[
            ReproExperiment(name="exp_a", description="A", overrides={"scale": 1.0}),
            ReproExperiment(name="exp_b", description="B", overrides={"scale": 2.0}),
        ],
    )

    def _factory(seed: int, exp: ReproExperiment):
        return DummyTrainer(seed=seed, scale=float(exp.overrides.get("scale", 1.0)))

    out = run_repro_suite(cfg, trainer_factory=_factory)
    assert out["n_runs"] == 4.0
    assert Path(out["summary_csv"]).exists()
    assert Path(out["summary_json"]).exists()
