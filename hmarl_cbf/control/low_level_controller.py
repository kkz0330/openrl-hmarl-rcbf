from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - import-safe fallback
    torch = None  # type: ignore[assignment]

from hmarl_cbf.control.constraint_builder import ConstraintBuilder
from hmarl_cbf.control.qp_solver import DifferentiableQPSolver
from hmarl_cbf.types import AgentObsLow, AgentState, QPParam, QPProblem, QPSolution


@dataclass(slots=True)
class LowLevelControlOutput:
    agent_id: int
    skill_id: int
    qp_param: QPParam
    fused_u_ref: np.ndarray
    fused_f_lin: np.ndarray
    problem: QPProblem
    solution: QPSolution
    safety_constraints: Dict[str, Any]
    neighbors_used: list[AgentState]
    obstacles_used: list[Dict[str, np.ndarray | float]]


class LowLevelSafeController:
    """
    Step-6 low-level control chain:
    obs_low + skill -> QP params -> fuse u_ref with skill reference -> CBF/CLF-QP -> safe action.
    """

    def __init__(
        self,
        low_policy: Any,
        constraint_builder: ConstraintBuilder,
        qp_solver: DifferentiableQPSolver,
        skill_ref_weight: float = 0.7,
        neighbor_perception_radius: float | None = None,
        obstacle_perception_range: float | None = None,
    ) -> None:
        if not (0.0 <= skill_ref_weight <= 1.0):
            raise ValueError("skill_ref_weight must be in [0, 1]")
        if neighbor_perception_radius is not None and neighbor_perception_radius < 0:
            raise ValueError("neighbor_perception_radius must be >= 0")
        if obstacle_perception_range is not None and obstacle_perception_range < 0:
            raise ValueError("obstacle_perception_range must be >= 0")
        self.low_policy = low_policy
        self.constraint_builder = constraint_builder
        self.qp_solver = qp_solver
        self.skill_ref_weight = float(skill_ref_weight)
        self.neighbor_perception_radius = neighbor_perception_radius
        self.obstacle_perception_range = obstacle_perception_range

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    def _infer_qp_param(self, obs_low: AgentObsLow, skill_id: int) -> QPParam:
        if self.low_policy is None:
            raise RuntimeError("low_policy is not set")

        if torch is not None and hasattr(self.low_policy, "forward"):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_low.flat, dtype=torch.float32).unsqueeze(0)
                skill_t = torch.as_tensor([skill_id], dtype=torch.long)
                qp_param = self.low_policy(obs_t, skill_t)
            return QPParam(
                u_ref=self._to_numpy(qp_param.u_ref).reshape(-1),
                r_diag=self._to_numpy(qp_param.r_diag).reshape(-1),
                w_clf=self._to_numpy(qp_param.w_clf).reshape(-1),
                cbf_k0=self._to_numpy(qp_param.cbf_k0).reshape(-1),
                cbf_k1=self._to_numpy(qp_param.cbf_k1).reshape(-1),
                clf_k=self._to_numpy(qp_param.clf_k).reshape(-1),
                f_lin=None,
                hocbf_gamma_h=(
                    self._to_numpy(qp_param.hocbf_gamma_h).reshape(-1)
                    if getattr(qp_param, "hocbf_gamma_h", None) is not None
                    else None
                ),
                hocbf_gamma_hdot=(
                    self._to_numpy(qp_param.hocbf_gamma_hdot).reshape(-1)
                    if getattr(qp_param, "hocbf_gamma_hdot", None) is not None
                    else None
                ),
            )

        qp_param = self.low_policy(obs_low, skill_id)
        if not isinstance(qp_param, QPParam):
            raise TypeError("low_policy callable must return QPParam")
        return QPParam(
            u_ref=self._to_numpy(qp_param.u_ref).reshape(-1),
            r_diag=self._to_numpy(qp_param.r_diag).reshape(-1),
            w_clf=self._to_numpy(qp_param.w_clf).reshape(-1),
            cbf_k0=self._to_numpy(qp_param.cbf_k0).reshape(-1),
            cbf_k1=self._to_numpy(qp_param.cbf_k1).reshape(-1),
            clf_k=self._to_numpy(qp_param.clf_k).reshape(-1),
            f_lin=None,
            hocbf_gamma_h=(
                self._to_numpy(qp_param.hocbf_gamma_h).reshape(-1)
                if getattr(qp_param, "hocbf_gamma_h", None) is not None
                else None
            ),
            hocbf_gamma_hdot=(
                self._to_numpy(qp_param.hocbf_gamma_hdot).reshape(-1)
                if getattr(qp_param, "hocbf_gamma_hdot", None) is not None
                else None
            ),
        )

    def _fuse_u_ref(self, policy_u_ref: np.ndarray, skill_u_ref: np.ndarray) -> np.ndarray:
        fused = self.skill_ref_weight * skill_u_ref + (1.0 - self.skill_ref_weight) * policy_u_ref
        return fused.astype(np.float32)

    def _filter_neighbors(self, state_i: AgentState, neighbors: list[AgentState]) -> list[AgentState]:
        if self.neighbor_perception_radius is None:
            return neighbors
        radius = float(self.neighbor_perception_radius)
        return [
            state_j
            for state_j in neighbors
            if float(np.linalg.norm(state_i.position - state_j.position)) <= radius
        ]

    def _filter_obstacles(
        self,
        state_i: AgentState,
        obstacles: list[Dict[str, np.ndarray | float]],
        obs_low: AgentObsLow,
    ) -> list[Dict[str, np.ndarray | float]]:
        max_range = float(self.obstacle_perception_range) if self.obstacle_perception_range is not None else float(obs_low.lidar_scan.max_range)
        perceived: list[Dict[str, np.ndarray | float]] = []
        for obs in obstacles:
            center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
            radius = float(obs["radius"])
            surface_distance = float(np.linalg.norm(center - state_i.position) - radius)
            if surface_distance <= max_range:
                perceived.append(obs)
        return perceived

    def solve_for_agent(
        self,
        agent_id: int,
        state_i: AgentState,
        neighbors: list[AgentState],
        obstacles: list[Dict[str, np.ndarray | float]],
        obs_low: AgentObsLow,
        skill_id: int,
        skill_u_ref: np.ndarray,
        safety_constraints: Dict[str, Any] | None = None,
        qp_param_override: QPParam | None = None,
        policy_u_ref_override: np.ndarray | None = None,
        fused_u_ref_override: np.ndarray | None = None,
    ) -> LowLevelControlOutput:
        qp_param = qp_param_override if qp_param_override is not None else self._infer_qp_param(obs_low=obs_low, skill_id=skill_id)
        if fused_u_ref_override is not None:
            fused_u_ref = np.asarray(fused_u_ref_override, dtype=np.float32).reshape(2)
        else:
            policy_u_ref = (
                np.asarray(policy_u_ref_override, dtype=np.float32).reshape(2)
                if policy_u_ref_override is not None
                else np.asarray(qp_param.u_ref, dtype=np.float32).reshape(2)
            )
            fused_u_ref = self._fuse_u_ref(
                policy_u_ref=policy_u_ref,
                skill_u_ref=np.asarray(skill_u_ref, dtype=np.float32).reshape(2),
            )
        fused_f_lin = -(
            np.asarray(qp_param.r_diag, dtype=np.float32).reshape(2)
            * np.asarray(fused_u_ref, dtype=np.float32).reshape(2)
        )
        problem = self.constraint_builder.build_for_agent(
            state_i=state_i,
            neighbors=neighbors,
            obstacles=obstacles,
            qp_param=qp_param,
            u_ref_override=fused_u_ref,
            f_lin_override=None,
            constraint_overrides=safety_constraints,
        )
        solution = self.qp_solver.solve(problem)
        return LowLevelControlOutput(
            agent_id=agent_id,
            skill_id=skill_id,
            qp_param=qp_param,
            fused_u_ref=fused_u_ref,
            fused_f_lin=fused_f_lin,
            problem=problem,
            solution=solution,
            safety_constraints=dict(safety_constraints or {}),
            neighbors_used=[
                AgentState(
                    agent_id=int(s.agent_id),
                    position=np.asarray(s.position, dtype=np.float32).copy(),
                    velocity=np.asarray(s.velocity, dtype=np.float32).copy(),
                    goal=np.asarray(s.goal, dtype=np.float32).copy(),
                    radius=float(s.radius),
                )
                for s in neighbors
            ],
            obstacles_used=[
                {
                    "center": np.asarray(o["center"], dtype=np.float32).reshape(2).copy(),
                    "radius": float(o["radius"]),
                }
                for o in obstacles
            ],
        )

    def solve_batch(
        self,
        states: Mapping[int, AgentState],
        obs_low: Mapping[int, AgentObsLow],
        skill_targets: Mapping[int, Dict[str, Any]],
        obstacles: list[Dict[str, np.ndarray | float]],
        qp_param_overrides: Mapping[int, QPParam] | None = None,
        policy_u_ref_overrides: Mapping[int, np.ndarray] | None = None,
        fused_u_ref_overrides: Mapping[int, np.ndarray] | None = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, LowLevelControlOutput]]:
        actions: Dict[int, np.ndarray] = {}
        outputs: Dict[int, LowLevelControlOutput] = {}
        qp_param_overrides = dict(qp_param_overrides or {})
        policy_u_ref_overrides = dict(policy_u_ref_overrides or {})
        fused_u_ref_overrides = dict(fused_u_ref_overrides or {})
        for agent_id, state_i in states.items():
            if agent_id not in skill_targets:
                raise KeyError(f"missing skill target for agent {agent_id}")
            target = skill_targets[agent_id]
            skill_id = int(target["skill_id"])
            skill_u_ref = np.asarray(target["u_ref_skill"], dtype=np.float32).reshape(2)
            safety_constraints = dict(target.get("safety_constraints", {}))

            neighbors_all = [state_j for other_id, state_j in states.items() if other_id != agent_id]
            neighbors = self._filter_neighbors(state_i=state_i, neighbors=neighbors_all)
            obstacles_local = self._filter_obstacles(state_i=state_i, obstacles=obstacles, obs_low=obs_low[agent_id])
            out = self.solve_for_agent(
                agent_id=agent_id,
                state_i=state_i,
                neighbors=neighbors,
                obstacles=obstacles_local,
                obs_low=obs_low[agent_id],
                skill_id=skill_id,
                skill_u_ref=skill_u_ref,
                safety_constraints=safety_constraints,
                qp_param_override=qp_param_overrides.get(agent_id, None),
                policy_u_ref_override=policy_u_ref_overrides.get(agent_id, None),
                fused_u_ref_override=fused_u_ref_overrides.get(agent_id, None),
            )
            actions[agent_id] = np.asarray(out.solution.action, dtype=np.float32).reshape(2)
            outputs[agent_id] = out
        return actions, outputs
