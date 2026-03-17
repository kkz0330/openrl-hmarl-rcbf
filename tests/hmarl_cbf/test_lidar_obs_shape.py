import numpy as np

from hmarl_cbf.env import LidarModel, ObservationBuilder
from hmarl_cbf.types import AgentState


def test_lidar_shape_and_normalization() -> None:
    lidar = LidarModel(n_beam=16, max_range=6.0, noise_std=0.0)
    scan = lidar.scan(origin=np.array([0.0, 0.0], dtype=np.float32), obstacles=[])
    assert scan.ranges.shape == (16,)
    assert np.allclose(scan.normalized, 1.0)


def test_observation_builder_shapes() -> None:
    lidar = LidarModel(n_beam=8, max_range=5.0, noise_std=0.0)
    builder = ObservationBuilder(lidar=lidar, neighbor_radius=4.0, max_neighbors=2)
    states = [
        AgentState(0, np.array([0.0, 0.0]), np.zeros(2), np.array([1.0, 0.0])),
        AgentState(1, np.array([1.0, 0.0]), np.zeros(2), np.array([2.0, 0.0])),
    ]
    obstacles = [{"center": np.array([2.0, 2.0], dtype=np.float32), "radius": 0.5}]
    obs = builder.build(states, obstacles)
    low = obs[0]["low"]
    assert low.lidar_scan.ranges.shape == (8,)
    assert low.neighbor_summary.shape == (8,)
    assert low.flat.ndim == 1
