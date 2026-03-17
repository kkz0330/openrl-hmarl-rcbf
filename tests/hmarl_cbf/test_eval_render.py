from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from hmarl_cbf.eval import EpisodeTrace, TrajectoryRenderer


def test_render_static_outputs_file() -> None:
    renderer = TrajectoryRenderer(world_size=5.0)
    positions = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.5, 0.2], [1.2, 0.2]],
            [[1.0, 0.5], [1.5, 0.3]],
        ],
        dtype=np.float32,
    )
    goals = np.array([[2.0, 1.0], [2.0, 0.5]], dtype=np.float32)
    trace = EpisodeTrace(
        positions=positions,
        goals=goals,
        obstacles=[{"center": np.array([0.0, 1.0], dtype=np.float32), "radius": 0.3}],
    )
    out_dir = Path("tests/.tmp_eval_render")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "traj.png"
    rendered = renderer.render_static(trace, out)
    assert Path(rendered).exists()
