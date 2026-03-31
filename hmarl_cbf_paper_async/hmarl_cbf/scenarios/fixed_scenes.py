from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


SceneDef = Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]


def _corridor_4uav_dual_passage(world_size: float, agent_radius: float) -> SceneDef:
    del world_size
    x_left = -5.6
    x_right = 5.6
    ys = [-2.2, 2.2]
    states: List[Dict[str, Any]] = []
    for y in ys:
        states.append(
            {
                "position": np.asarray([x_left, y], dtype=np.float32),
                "velocity": np.asarray([0.0, 0.0], dtype=np.float32),
                "goal": np.asarray([x_right, y], dtype=np.float32),
                "radius": float(agent_radius),
            }
        )
    for y in ys:
        states.append(
            {
                "position": np.asarray([x_right, y], dtype=np.float32),
                "velocity": np.asarray([0.0, 0.0], dtype=np.float32),
                "goal": np.asarray([x_left, y], dtype=np.float32),
                "radius": float(agent_radius),
            }
        )
    obstacles: List[Dict[str, Any]] = [
        {"center": np.asarray([0.0, -2.7], dtype=np.float32), "radius": 0.9},
        {"center": np.asarray([0.0, -0.9], dtype=np.float32), "radius": 0.9},
        {"center": np.asarray([0.0, 0.9], dtype=np.float32), "radius": 0.9},
        {"center": np.asarray([0.0, 2.7], dtype=np.float32), "radius": 0.9},
    ]
    return states, obstacles


def _square_open_trap_4uav(world_size: float, agent_radius: float) -> SceneDef:
    del world_size
    starts = [
        np.asarray([-0.6, 0.5], dtype=np.float32),
        np.asarray([0.6, 0.5], dtype=np.float32),
        np.asarray([-0.6, -0.3], dtype=np.float32),
        np.asarray([0.6, -0.3], dtype=np.float32),
    ]
    goals = [
        np.asarray([-0.8, -5.6], dtype=np.float32),
        np.asarray([0.8, -5.6], dtype=np.float32),
        np.asarray([-1.6, -5.0], dtype=np.float32),
        np.asarray([1.6, -5.0], dtype=np.float32),
    ]
    states: List[Dict[str, Any]] = []
    for pos, goal in zip(starts, goals):
        states.append(
            {
                "position": pos,
                "velocity": np.asarray([0.0, 0.0], dtype=np.float32),
                "goal": goal,
                "radius": float(agent_radius),
            }
        )
    obstacles: List[Dict[str, Any]] = [
        {"center": np.asarray([-1.8, -0.9], dtype=np.float32), "radius": 0.9},
        {"center": np.asarray([-1.8, 0.9], dtype=np.float32), "radius": 0.9},
        {"center": np.asarray([1.8, -0.9], dtype=np.float32), "radius": 0.9},
        {"center": np.asarray([1.8, 0.9], dtype=np.float32), "radius": 0.9},
        {"center": np.asarray([0.0, -1.9], dtype=np.float32), "radius": 0.9},
    ]
    return states, obstacles


def _u_trap_4uav(world_size: float, agent_radius: float) -> SceneDef:
    del world_size
    starts = [
        np.asarray([-0.7, 0.8], dtype=np.float32),
        np.asarray([0.7, 0.8], dtype=np.float32),
        np.asarray([-0.7, -0.1], dtype=np.float32),
        np.asarray([0.7, -0.1], dtype=np.float32),
    ]
    goals = [
        np.asarray([-1.0, 5.8], dtype=np.float32),
        np.asarray([1.0, 5.8], dtype=np.float32),
        np.asarray([-1.8, 5.2], dtype=np.float32),
        np.asarray([1.8, 5.2], dtype=np.float32),
    ]
    states: List[Dict[str, Any]] = []
    for pos, goal in zip(starts, goals):
        states.append(
            {
                "position": pos,
                "velocity": np.asarray([0.0, 0.0], dtype=np.float32),
                "goal": goal,
                "radius": float(agent_radius),
            }
        )
    obstacles: List[Dict[str, Any]] = [
        {"center": np.asarray([-1.8, 1.4], dtype=np.float32), "radius": 0.9},
        {"center": np.asarray([-1.8, -0.2], dtype=np.float32), "radius": 0.9},
        {"center": np.asarray([1.8, 1.4], dtype=np.float32), "radius": 0.9},
        {"center": np.asarray([1.8, -0.2], dtype=np.float32), "radius": 0.9},
        {"center": np.asarray([0.0, 2.6], dtype=np.float32), "radius": 0.9},
    ]
    return states, obstacles


def _narrow_gap_4uav(world_size: float, agent_radius: float) -> SceneDef:
    del world_size
    starts = [
        np.asarray([-5.4, -1.0], dtype=np.float32),
        np.asarray([-5.4, 1.0], dtype=np.float32),
        np.asarray([5.4, -1.0], dtype=np.float32),
        np.asarray([5.4, 1.0], dtype=np.float32),
    ]
    goals = [
        np.asarray([5.4, -1.0], dtype=np.float32),
        np.asarray([5.4, 1.0], dtype=np.float32),
        np.asarray([-5.4, -1.0], dtype=np.float32),
        np.asarray([-5.4, 1.0], dtype=np.float32),
    ]
    states: List[Dict[str, Any]] = []
    for pos, goal in zip(starts, goals):
        states.append(
            {
                "position": pos,
                "velocity": np.asarray([0.0, 0.0], dtype=np.float32),
                "goal": goal,
                "radius": float(agent_radius),
            }
        )
    obstacles: List[Dict[str, Any]] = [
        {"center": np.asarray([0.0, 1.15], dtype=np.float32), "radius": 0.85},
        {"center": np.asarray([0.0, -1.15], dtype=np.float32), "radius": 0.85},
        {"center": np.asarray([2.4, 2.0], dtype=np.float32), "radius": 0.75},
        {"center": np.asarray([2.4, -2.0], dtype=np.float32), "radius": 0.75},
    ]
    return states, obstacles


_SCENE_BUILDERS = {
    "corridor_4uav_dual_passage": _corridor_4uav_dual_passage,
    "square_open_trap_4uav": _square_open_trap_4uav,
    "u_trap_4uav": _u_trap_4uav,
    "narrow_gap_4uav": _narrow_gap_4uav,
}


def fixed_scene_names() -> List[str]:
    return sorted(_SCENE_BUILDERS.keys())


def build_fixed_scene(name: str, world_size: float, agent_radius: float) -> SceneDef:
    if name not in _SCENE_BUILDERS:
        raise KeyError(f"unknown fixed scene: {name}")
    return _SCENE_BUILDERS[name](world_size=float(world_size), agent_radius=float(agent_radius))
