from __future__ import annotations

from math import atan2
from typing import Any, Dict, List

import numpy as np

from hmarl_cbf.types import AgentObsLow, AgentState, SkillSpec

SKILL_TURN_LEFT = 0
SKILL_TURN_RIGHT = 1
SKILL_ACCELERATE = 2
SKILL_DECELERATE = 3
SKILL_CRUISE = 4


def _norm(vec: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.linalg.norm(vec) + eps)


def _unit(vec: np.ndarray) -> np.ndarray:
    n = _norm(vec)
    if n <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vec / n).astype(np.float32)


def _signed_angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > np.pi:
        d -= 2.0 * np.pi
    while d < -np.pi:
        d += 2.0 * np.pi
    return float(d)


def _state_speed(state: AgentState) -> float:
    return float(np.linalg.norm(state.velocity))


def _state_heading(state: AgentState) -> float:
    if np.linalg.norm(state.velocity) > 1e-6:
        return float(atan2(state.velocity[1], state.velocity[0]))
    goal_vec = state.goal - state.position
    return float(atan2(goal_vec[1], goal_vec[0]))


def _goal_dir_from_state(state: AgentState) -> np.ndarray:
    return _unit(state.goal - state.position)


def _goal_dir_from_obs(obs: AgentObsLow) -> np.ndarray:
    return _unit(obs.goal_relative)


def _clip_action(action: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
    if "u_min" in ctx and "u_max" in ctx:
        u_min = np.asarray(ctx["u_min"], dtype=np.float32).reshape(2)
        u_max = np.asarray(ctx["u_max"], dtype=np.float32).reshape(2)
        return np.clip(action, u_min, u_max).astype(np.float32)
    limit = float(ctx.get("action_limit", 1.0))
    return np.clip(action, -limit, limit).astype(np.float32)


def _state_vec_speed_heading_goal(s_i: np.ndarray, ctx: Dict[str, Any]) -> tuple[float, float, np.ndarray, float]:
    s_i = np.asarray(s_i, dtype=np.float32).reshape(-1)
    vel = s_i[2:4] if s_i.shape[0] >= 4 else np.zeros(2, dtype=np.float32)
    speed = float(np.linalg.norm(vel))
    heading = float(atan2(float(vel[1]), float(vel[0]))) if speed > 1e-6 else float(ctx.get("heading_ref", 0.0))
    if s_i.shape[0] >= 6:
        goal_rel = s_i[4:6]
    else:
        goal_rel = np.asarray(ctx.get("goal_relative", np.array([1.0, 0.0], dtype=np.float32)), dtype=np.float32).reshape(2)
    goal_dir = _unit(goal_rel)
    goal_dist = float(np.linalg.norm(goal_rel))
    return speed, heading, goal_dir, goal_dist


def _default_initiation(_: AgentState, __: Dict[str, Any]) -> bool:
    return True


def _turn_initiation(state: AgentState, ctx: Dict[str, Any]) -> bool:
    return _state_speed(state) >= float(ctx.get("turn_min_speed", 0.2))


def _accelerate_initiation(state: AgentState, ctx: Dict[str, Any]) -> bool:
    return _state_speed(state) <= float(ctx.get("accelerate_init_max_speed", 5.0))


def _decelerate_initiation(state: AgentState, ctx: Dict[str, Any]) -> bool:
    min_speed = float(ctx.get("decelerate_init_min_speed", ctx.get("goal_speed_threshold", 0.1)))
    return _state_speed(state) >= min_speed


def _cruise_initiation(state: AgentState, ctx: Dict[str, Any]) -> bool:
    return _state_speed(state) >= float(ctx.get("cruise_min_speed", 0.2))


def _goal_reached(state: AgentState, ctx: Dict[str, Any]) -> bool:
    dist_threshold = float(ctx.get("goal_threshold", 0.3))
    speed_threshold = float(ctx.get("goal_speed_threshold", 0.1))
    dist_ok = float(np.linalg.norm(state.goal - state.position)) <= dist_threshold
    speed_ok = _state_speed(state) <= speed_threshold
    return bool(dist_ok and speed_ok)


def _turn_termination_set(state: AgentState, ctx: Dict[str, Any]) -> bool:
    if _goal_reached(state, ctx):
        return True
    if "target_heading" not in ctx:
        return False
    heading = _state_heading(state)
    err = abs(_signed_angle_diff(heading, float(ctx["target_heading"])))
    return bool(err <= float(ctx.get("turn_heading_tol", 0.2)))


def _accelerate_termination_set(state: AgentState, ctx: Dict[str, Any]) -> bool:
    if _goal_reached(state, ctx):
        return True
    speed = _state_speed(state)
    mode = str(ctx.get("accelerate_mode", "goal_tracking")).strip().lower()
    if mode in {"hmarl_like", "along_velocity", "velocity_increment"}:
        delta_speed = float(max(ctx.get("accelerate_delta_speed", 0.3), 0.0))
        start_speed = float(ctx.get("start_speed", speed))
        target_speed = start_speed + delta_speed
        tol = float(ctx.get("accelerate_stop_eps", 0.05))
        return speed >= max(0.0, target_speed - tol)
    return speed >= float(ctx.get("target_speed", 1.2))


def _decelerate_termination_set(state: AgentState, ctx: Dict[str, Any]) -> bool:
    if _goal_reached(state, ctx):
        return True
    speed = _state_speed(state)
    mode = str(ctx.get("decelerate_mode", "zero_track")).strip().lower()
    if mode in {"hmarl_like", "along_velocity", "velocity_decrement"}:
        delta_speed = float(max(ctx.get("decelerate_delta_speed", 0.2), 0.0))
        start_speed = float(ctx.get("start_speed", speed))
        target_speed = max(0.0, start_speed - delta_speed)
        tol = float(ctx.get("decelerate_stop_eps", 0.05))
        return speed <= (target_speed + tol)
    return speed <= float(ctx.get("decelerate_target_speed", 0.3))


def _cruise_termination_set(state: AgentState, ctx: Dict[str, Any]) -> bool:
    return _goal_reached(state, ctx)


def _termination_with_timeout(
    state: AgentState,
    ctx: Dict[str, Any],
    tau: int,
    termination_set_fn,
) -> bool:
    return bool(termination_set_fn(state, ctx) or tau >= int(ctx.get("max_duration", 20)))


def _safety_constraints_common(_: AgentState, ctx: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cbf_mode": str(ctx.get("cbf_mode", "distributed_ecbf")),
        "clf_mode": "soft",
        "d_min_agent": float(ctx.get("d_min_agent", 0.6)),
        "d_safe_obs": float(ctx.get("d_safe_obs", 0.6)),
        "boundary_cbf": bool(ctx.get("boundary_cbf", False)),
        "world_size": float(ctx.get("world_size", 0.0)),
        "boundary_margin": float(ctx.get("boundary_margin", 0.0)),
        "cbf_u_max": float(ctx.get("cbf_u_max", ctx.get("action_limit", 1.0))),
        "cbf_share_agent": float(ctx.get("cbf_share_agent", 0.5)),
        "cbf_share_obs": float(ctx.get("cbf_share_obs", 1.0)),
        "cbf_eps": float(ctx.get("cbf_eps", 1e-4)),
        "robust_cbf": bool(ctx.get("robust_cbf", False)),
        "disturbance_accel_max": float(ctx.get("disturbance_accel_max", 0.0)),
        "relative_disturbance_accel_max": float(ctx.get("relative_disturbance_accel_max", 0.0)),
        "use_input_bounds": bool(ctx.get("use_input_bounds", True)),
        "slow_radius": float(ctx.get("slow_radius", 1.5)),
        "goal_stop_min_speed": float(ctx.get("goal_stop_min_speed", 0.0)),
        "rect_base_margin_extra": float(ctx.get("rect_base_margin_extra", 0.0)),
        "rect_corner_margin_enabled": bool(ctx.get("rect_corner_margin_enabled", False)),
        "rect_corner_margin_max": float(ctx.get("rect_corner_margin_max", 0.0)),
        "rect_corner_proximity_distance": float(ctx.get("rect_corner_proximity_distance", 0.4)),
        "rect_corner_speed_min": float(ctx.get("rect_corner_speed_min", 0.05)),
        "rect_corner_alignment_power": float(ctx.get("rect_corner_alignment_power", 1.0)),
        "rect_dual_edge_cbf_enabled": bool(ctx.get("rect_dual_edge_cbf_enabled", False)),
        "rect_dual_edge_proximity_distance": float(ctx.get("rect_dual_edge_proximity_distance", 0.0)),
        "rect_smooth_tau": float(ctx.get("rect_smooth_tau", 0.1)),
    }


def _turn_constraints(state: AgentState, ctx: Dict[str, Any]) -> Dict[str, Any]:
    out = _safety_constraints_common(state, ctx)
    out["turn_rate_weight"] = float(ctx.get("turn_rate_weight", 1.0))
    theta_des = float(ctx.get("target_heading", _state_heading(state)))
    keep_speed = bool(ctx.get("turn_keep_speed", True))
    if keep_speed:
        vmag = float(ctx.get("turn_speed_ref", ctx.get("start_speed", _state_speed(state))))
    else:
        vmag = float(ctx.get("turn_vmag", ctx.get("ref_speed", 0.8)))
        slow_radius = float(ctx.get("slow_radius", 0.0))
        if slow_radius > 0.0:
            goal_dist = float(np.linalg.norm(state.goal - state.position))
            speed_scale = float(np.clip(goal_dist / slow_radius, 0.0, 1.0))
            vmag = max(float(ctx.get("goal_stop_min_speed", 0.0)), vmag * speed_scale)
    out["clf_v_des_vector"] = np.asarray(
        [vmag * np.cos(theta_des), vmag * np.sin(theta_des)],
        dtype=np.float32,
    )
    return out


def _accelerate_constraints(state: AgentState, ctx: Dict[str, Any]) -> Dict[str, Any]:
    out = _safety_constraints_common(state, ctx)
    out["speed_cap"] = float(ctx.get("accelerate_speed_cap", 2.0))
    return out


def _decelerate_constraints(state: AgentState, ctx: Dict[str, Any]) -> Dict[str, Any]:
    out = _safety_constraints_common(state, ctx)
    out["brake_bias"] = float(ctx.get("brake_bias", 1.0))
    out["clf_v_des_speed"] = float(ctx.get("decelerate_clf_speed", 0.0))
    return out


def _cruise_constraints(state: AgentState, ctx: Dict[str, Any]) -> Dict[str, Any]:
    out = _safety_constraints_common(state, ctx)
    out["cruise_ref_speed"] = float(ctx.get("cruise_ref_speed", ctx.get("start_speed", ctx.get("ref_speed", 0.8))))
    return out


def _intrinsic_reward_common(s_i: np.ndarray, a_i: np.ndarray, ctx: Dict[str, Any]) -> float:
    speed, heading, _, _ = _state_vec_speed_heading_goal(s_i, ctx)
    a_i = np.asarray(a_i, dtype=np.float32).reshape(2)
    accel_pen = float(ctx.get("w_accel", 0.05)) * float(np.dot(a_i, a_i))

    heading_dir = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
    lateral = float(np.cross(np.append(heading_dir, 0.0), np.append(a_i, 0.0))[2])
    turn_pen = float(ctx.get("w_turn", 0.03)) * abs(lateral)

    ref_speed = float(ctx.get("cruise_ref_speed", ctx.get("start_speed", ctx.get("ref_speed", 0.8))))
    speed_pen = float(ctx.get("w_speed_dev", 0.04)) * abs(speed - ref_speed)

    heading_ref = float(ctx.get("heading_ref", heading))
    heading_pen = float(ctx.get("w_heading_dev", 0.02)) * abs(_signed_angle_diff(heading, heading_ref))
    return float(-(accel_pen + turn_pen + speed_pen + heading_pen))


def _turn_left_policy(obs: AgentObsLow, ctx: Dict[str, Any]) -> np.ndarray:
    vel = obs.self_state[2:4] if obs.self_state.shape[0] >= 4 else np.zeros(2, dtype=np.float32)
    theta_des = float(ctx.get("target_heading", atan2(float(vel[1]), float(vel[0]))))
    keep_speed = bool(ctx.get("turn_keep_speed", True))
    if keep_speed:
        vmag = float(ctx.get("turn_speed_ref", ctx.get("start_speed", float(np.linalg.norm(vel)))))
    else:
        vmag = float(ctx.get("turn_vmag", ctx.get("ref_speed", 0.8)))
    v_des = np.asarray([vmag * np.cos(theta_des), vmag * np.sin(theta_des)], dtype=np.float32)
    turn_track_kp = float(ctx.get("turn_track_kp", 1.0))
    action = turn_track_kp * (v_des - vel)
    return _clip_action(action, ctx)


def _turn_right_policy(obs: AgentObsLow, ctx: Dict[str, Any]) -> np.ndarray:
    vel = obs.self_state[2:4] if obs.self_state.shape[0] >= 4 else np.zeros(2, dtype=np.float32)
    theta_des = float(ctx.get("target_heading", atan2(float(vel[1]), float(vel[0]))))
    keep_speed = bool(ctx.get("turn_keep_speed", True))
    if keep_speed:
        vmag = float(ctx.get("turn_speed_ref", ctx.get("start_speed", float(np.linalg.norm(vel)))))
    else:
        vmag = float(ctx.get("turn_vmag", ctx.get("ref_speed", 0.8)))
    v_des = np.asarray([vmag * np.cos(theta_des), vmag * np.sin(theta_des)], dtype=np.float32)
    turn_track_kp = float(ctx.get("turn_track_kp", 1.0))
    action = turn_track_kp * (v_des - vel)
    return _clip_action(action, ctx)


def _accelerate_policy(obs: AgentObsLow, ctx: Dict[str, Any]) -> np.ndarray:
    vel = obs.self_state[2:4] if obs.self_state.shape[0] >= 4 else np.zeros(2, dtype=np.float32)
    speed = float(np.linalg.norm(vel))
    goal_dir = _goal_dir_from_obs(obs)
    mode = str(ctx.get("accelerate_mode", "goal_tracking")).strip().lower()
    if mode in {"hmarl_like", "along_velocity", "velocity_increment"}:
        delta_speed = float(max(ctx.get("accelerate_delta_speed", 0.3), 0.0))
        start_speed = float(ctx.get("start_speed", speed))
        target_speed = start_speed + delta_speed
        dv = max(0.0, target_speed - speed)
        if speed > 1e-4:
            move_dir = _unit(vel)
        else:
            heading = float(ctx.get("heading_ref", atan2(float(goal_dir[1]), float(goal_dir[0]))))
            move_dir = np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float32)
        if dv <= 0.0:
            action = np.zeros(2, dtype=np.float32)
        else:
            dt = float(ctx.get("dt", 0.0))
            if dt > 1e-8:
                # Delta-speed semantics: increase speed by at most `accelerate_delta_speed`
                # in one step, and clamp to target speed.
                action = (dv / dt) * move_dir
            else:
                accel_step = float(ctx.get("accelerate_step", ctx.get("accelerate_gain", 0.8)))
                action = accel_step * move_dir
        return _clip_action(action, ctx)

    vel_dir = _unit(vel) if speed > 1e-4 else goal_dir
    heading_blend = float(ctx.get("accelerate_heading_blend", 0.0))
    heading_blend = float(np.clip(heading_blend, 0.0, 1.0))
    move_dir = _unit((1.0 - heading_blend) * goal_dir + heading_blend * vel_dir)
    target_speed = float(max(ctx.get("target_speed", 1.2), float(ctx.get("start_speed", speed)) + float(ctx.get("accelerate_delta_speed", 0.4))))
    kp = float(ctx.get("accelerate_speed_kp", 1.2))
    action = kp * (target_speed - speed) * move_dir
    return _clip_action(action, ctx)


def _decelerate_policy(obs: AgentObsLow, ctx: Dict[str, Any]) -> np.ndarray:
    vel = obs.self_state[2:4] if obs.self_state.shape[0] >= 4 else np.zeros(2, dtype=np.float32)
    speed = float(np.linalg.norm(vel))
    stop_eps = float(ctx.get("decelerate_stop_eps", 0.05))
    if speed <= stop_eps:
        return np.zeros(2, dtype=np.float32)

    mode = str(ctx.get("decelerate_mode", "zero_track")).strip().lower()
    if mode in {"hmarl_like", "along_velocity", "velocity_decrement"}:
        delta_speed = float(max(ctx.get("decelerate_delta_speed", 0.2), 0.0))
        dv = min(delta_speed, speed)
        if speed <= 1e-6 or dv <= 0.0:
            action = np.zeros(2, dtype=np.float32)
        else:
            dt = float(ctx.get("dt", 0.0))
            decel_dir = _unit(vel)
            if dt > 1e-8:
                # Enforce speed decrement semantics:
                # target speed = max(0, speed - delta_speed) in one environment step.
                action = -(dv / dt) * decel_dir
            else:
                decel_step = float(ctx.get("decelerate_step", ctx.get("decelerate_gain", 0.8)))
                action = -decel_step * decel_dir
    else:
        kv = float(ctx.get("decelerate_kv", 1.5))
        action = -kv * vel
    return _clip_action(action, ctx)


def _cruise_policy(obs: AgentObsLow, ctx: Dict[str, Any]) -> np.ndarray:
    vel = obs.self_state[2:4] if obs.self_state.shape[0] >= 4 else np.zeros(2, dtype=np.float32)
    speed = float(np.linalg.norm(vel))
    goal_dir = _goal_dir_from_obs(obs)
    ref_speed = float(ctx.get("cruise_ref_speed", ctx.get("start_speed", ctx.get("ref_speed", speed))))
    kp = float(ctx.get("cruise_speed_kp", 0.8))
    k_align = float(ctx.get("cruise_align_kp", 0.3))
    accel_long = kp * (ref_speed - speed) * goal_dir
    heading_dir = _unit(vel if speed > 1e-6 else goal_dir)
    lateral_err = np.array([-heading_dir[1], heading_dir[0]], dtype=np.float32)
    align = k_align * np.dot(goal_dir, lateral_err) * lateral_err
    return _clip_action(accel_long + align, ctx)


def _turn_intrinsic_reward(s_i: np.ndarray, a_i: np.ndarray, ctx: Dict[str, Any]) -> float:
    base = _intrinsic_reward_common(s_i, a_i, ctx)
    return float(base - float(ctx.get("w_turn_extra", 0.02)) * float(np.linalg.norm(a_i)))


def _accelerate_intrinsic_reward(s_i: np.ndarray, a_i: np.ndarray, ctx: Dict[str, Any]) -> float:
    base = _intrinsic_reward_common(s_i, a_i, ctx)
    speed, _, _, _ = _state_vec_speed_heading_goal(s_i, ctx)
    mode = str(ctx.get("accelerate_mode", "goal_tracking")).strip().lower()
    if mode in {"hmarl_like", "along_velocity", "velocity_increment"}:
        delta_speed = float(max(ctx.get("accelerate_delta_speed", 0.3), 0.0))
        start_speed = float(ctx.get("start_speed", speed))
        target = start_speed + delta_speed
        bonus = float(ctx.get("w_accel_target", 0.03)) * float(np.exp(-abs(speed - target)))
    else:
        bonus = float(ctx.get("w_accel_target", 0.03)) * min(speed, float(ctx.get("target_speed", 1.2)))
    return float(base + bonus)


def _decelerate_intrinsic_reward(s_i: np.ndarray, a_i: np.ndarray, ctx: Dict[str, Any]) -> float:
    base = _intrinsic_reward_common(s_i, a_i, ctx)
    speed, _, _, _ = _state_vec_speed_heading_goal(s_i, ctx)
    mode = str(ctx.get("decelerate_mode", "zero_track")).strip().lower()
    if mode in {"hmarl_like", "along_velocity", "velocity_decrement"}:
        delta_speed = float(max(ctx.get("decelerate_delta_speed", 0.2), 0.0))
        start_speed = float(ctx.get("start_speed", speed))
        target = max(0.0, start_speed - delta_speed)
    else:
        target = float(ctx.get("decelerate_target_speed", 0.3))
    bonus = float(ctx.get("w_decel_target", 0.04)) * float(np.exp(-abs(speed - target)))
    return float(base + bonus)


def _cruise_intrinsic_reward(s_i: np.ndarray, a_i: np.ndarray, ctx: Dict[str, Any]) -> float:
    base = _intrinsic_reward_common(s_i, a_i, ctx)
    speed, _, _, _ = _state_vec_speed_heading_goal(s_i, ctx)
    ref = float(ctx.get("cruise_ref_speed", ctx.get("start_speed", ctx.get("ref_speed", 0.8))))
    bonus = float(ctx.get("w_cruise_stable", 0.05)) * np.exp(-abs(speed - ref))
    return float(base + bonus)


def build_default_skill_library(max_duration: int = 20) -> List[SkillSpec]:
    skills = [
        SkillSpec(
            skill_id=SKILL_TURN_LEFT,
            name="turn_left",
            initiation_set_fn=_turn_initiation,
            termination_set_fn=_turn_termination_set,
            max_duration=max_duration,
            termination_fn=lambda s, c, tau: _termination_with_timeout(s, c, tau, _turn_termination_set),
            safety_constraints_fn=_turn_constraints,
            intrinsic_reward_fn=_turn_intrinsic_reward,
            safe_skill_policy=_turn_left_policy,
        ),
        SkillSpec(
            skill_id=SKILL_TURN_RIGHT,
            name="turn_right",
            initiation_set_fn=_turn_initiation,
            termination_set_fn=_turn_termination_set,
            max_duration=max_duration,
            termination_fn=lambda s, c, tau: _termination_with_timeout(s, c, tau, _turn_termination_set),
            safety_constraints_fn=_turn_constraints,
            intrinsic_reward_fn=_turn_intrinsic_reward,
            safe_skill_policy=_turn_right_policy,
        ),
        SkillSpec(
            skill_id=SKILL_ACCELERATE,
            name="accelerate",
            initiation_set_fn=_accelerate_initiation,
            termination_set_fn=_accelerate_termination_set,
            max_duration=max_duration,
            termination_fn=lambda s, c, tau: _termination_with_timeout(s, c, tau, _accelerate_termination_set),
            safety_constraints_fn=_accelerate_constraints,
            intrinsic_reward_fn=_accelerate_intrinsic_reward,
            safe_skill_policy=_accelerate_policy,
        ),
        SkillSpec(
            skill_id=SKILL_DECELERATE,
            name="decelerate",
            initiation_set_fn=_decelerate_initiation,
            termination_set_fn=_decelerate_termination_set,
            max_duration=max_duration,
            termination_fn=lambda s, c, tau: _termination_with_timeout(s, c, tau, _decelerate_termination_set),
            safety_constraints_fn=_decelerate_constraints,
            intrinsic_reward_fn=_decelerate_intrinsic_reward,
            safe_skill_policy=_decelerate_policy,
        ),
        SkillSpec(
            skill_id=SKILL_CRUISE,
            name="cruise",
            initiation_set_fn=_cruise_initiation,
            termination_set_fn=_cruise_termination_set,
            max_duration=max_duration,
            termination_fn=lambda s, c, tau: _termination_with_timeout(s, c, tau, _cruise_termination_set),
            safety_constraints_fn=_cruise_constraints,
            intrinsic_reward_fn=_cruise_intrinsic_reward,
            safe_skill_policy=_cruise_policy,
        ),
    ]
    return skills
