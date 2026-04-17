from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _find_gcbfplus_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "_ext" / "gcbfplus"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate _ext/gcbfplus from the current workspace")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample GCBF+ DoubleIntegrator scenes and export them as HMARL-CBF fixed scenes."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/hmarl_cbf_scene_library/gcbfplus_rect_scenes.json",
        help="Output JSON path.",
    )
    parser.add_argument("--n-scenes", type=int, default=20, help="Number of scenes to export.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents in each scene.")
    parser.add_argument("--area-size", type=float, default=10.0, help="GCBF+ sampling area size.")
    parser.add_argument("--max-step", type=int, default=256)
    parser.add_argument("--max-travel", type=float, default=0.0, help="0 means use GCBF+ default (None).")
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--n-obs", type=int, default=8)
    parser.add_argument("--obs-len-min", type=float, default=0.1)
    parser.add_argument("--obs-len-max", type=float, default=0.5)
    parser.add_argument(
        "--agent-radius",
        type=float,
        default=0.05,
        help="Radius used in exported HMARL-CBF state objects.",
    )
    parser.add_argument(
        "--center-coordinates",
        action="store_true",
        help="Shift GCBF+ [0, area_size] coordinates to a centered frame by subtracting area_size / 2.",
    )
    return parser.parse_args()


def _to_float_list(values: Any) -> List[float]:
    return [float(v) for v in values]


def _convert_scene(graph: Any, area_size: float, agent_radius: float, center_coordinates: bool) -> Dict[str, Any]:
    shift = area_size / 2.0 if center_coordinates else 0.0
    env_states = graph.env_states
    agents = env_states.agent
    goals = env_states.goal
    obstacles = env_states.obstacle

    states: List[Dict[str, Any]] = []
    for idx in range(int(agents.shape[0])):
        pos = agents[idx, :2]
        vel = agents[idx, 2:4]
        goal = goals[idx, :2]
        if center_coordinates:
            pos = pos - shift
            goal = goal - shift
        states.append(
            {
                "position": _to_float_list(pos),
                "velocity": _to_float_list(vel),
                "goal": _to_float_list(goal),
                "radius": float(agent_radius),
            }
        )

    obstacle_list: List[Dict[str, Any]] = []
    for idx in range(int(obstacles.center.shape[0])):
        center = obstacles.center[idx]
        if center_coordinates:
            center = center - shift
        obstacle_list.append(
            {
                "type": "rect",
                "center": _to_float_list(center),
                "half_extents": [
                    float(obstacles.width[idx]) / 2.0,
                    float(obstacles.height[idx]) / 2.0,
                ],
                "yaw": float(obstacles.theta[idx]),
            }
        )

    return {
        "states": states,
        "obstacles": obstacle_list,
    }


def main() -> None:
    args = _parse_args()
    gcbfplus_root = _find_gcbfplus_root()
    sys.path.insert(0, str(gcbfplus_root))

    import jax.random as jr  # type: ignore
    from gcbfplus.env.double_integrator import DoubleIntegrator  # type: ignore

    params = dict(DoubleIntegrator.PARAMS)
    params.update(
        {
            "n_obs": int(args.n_obs),
            "obs_len_range": [float(args.obs_len_min), float(args.obs_len_max)],
        }
    )
    max_travel = None if float(args.max_travel) <= 0.0 else float(args.max_travel)
    env = DoubleIntegrator(
        num_agents=int(args.num_agents),
        area_size=float(args.area_size),
        max_step=int(args.max_step),
        max_travel=max_travel,
        dt=float(args.dt),
        params=params,
    )

    scenes: List[Dict[str, Any]] = []
    for scene_idx in range(int(args.n_scenes)):
        key = jr.PRNGKey(int(args.seed) + scene_idx)
        graph = env.reset(key)
        scene = _convert_scene(
            graph=graph,
            area_size=float(args.area_size),
            agent_radius=float(args.agent_radius),
            center_coordinates=bool(args.center_coordinates),
        )
        scene["scene_id"] = f"gcbfplus_scene_{scene_idx:04d}"
        scene["source"] = "gcbfplus_double_integrator"
        scene["generator"] = {
            "num_agents": int(args.num_agents),
            "area_size": float(args.area_size),
            "n_obs": int(args.n_obs),
            "obs_len_range": [float(args.obs_len_min), float(args.obs_len_max)],
            "max_travel": None if max_travel is None else float(max_travel),
            "dt": float(args.dt),
            "center_coordinates": bool(args.center_coordinates),
        }
        scenes.append(scene)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "hmarl_cbf_fixed_scene_list",
        "source": "gcbfplus_double_integrator",
        "n_scenes": len(scenes),
        "scenes": scenes,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
