from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import yaml

from hmarl_cbf.openrl_compat import parse_openrl_default_config
from hmarl_cbf.openrl_agents import LowDiffQPAgent, LowDiffQPRolloutSample
from hmarl_cbf.openrl_envs import CoreEnv, HighLevelOpenRLEnv, HighLevelOpenRLEnvConfig, LowLevelOpenRLEnv, LowLevelOpenRLEnvConfig
from hmarl_cbf.openrl_models import HighLevelMAPPONet, LowQPActorNetwork, LowQPCriticNetwork, LowQPDecoder
from hmarl_cbf.skills import build_default_skill_library
from hmarl_cbf.types import AgentObsLow, AgentState


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict):
        raise ValueError("config root must be a mapping")
    return payload


def apply_config_section_defaults(
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    *,
    section: str,
    defaults: Mapping[str, Any],
) -> argparse.Namespace:
    section_payload = cfg.get(section, {})
    if section_payload is None:
        section_payload = {}
    if not isinstance(section_payload, Mapping):
        raise ValueError(f"config section '{section}' must be a mapping")

    for key, fallback in defaults.items():
        current = getattr(args, key)
        if current is None:
            value = section_payload.get(key, fallback)
            setattr(args, key, value)
    return args


def low_openrl_cfg(n_skills: int) -> Any:
    cfg = parse_openrl_default_config()
    cfg.n_skills = int(n_skills)
    cfg.use_naive_recurrent_policy = False
    cfg.use_recurrent_policy = False
    cfg.recurrent_N = 1
    cfg.use_fp16 = False
    cfg.use_deepspeed = False
    cfg.use_orthogonal = True
    cfg.rnn_type = "gru"
    cfg.gain = 0.01
    cfg.hidden_size = 128
    cfg.low_hidden_size = 128
    cfg.use_valuenorm = False
    cfg.use_policy_vhead = False
    return cfg


def build_core_env(cfg: Mapping[str, Any]) -> CoreEnv:
    return CoreEnv.from_env_config(cfg["env"])


def build_low_decoder_kwargs(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    low_qp_cfg = dict(cfg.get("low_level_qp", {}))
    skills_cfg = dict(cfg.get("skills", {}).get("params", {}))
    safety_cfg = dict(cfg.get("safety", {}))
    action_dim = int(cfg["model"].get("action_dim", 2))
    parameterize_cbf_constraints = bool(low_qp_cfg.get("parameterize_cbf_constraints", True))

    base_cbf_k0 = float(low_qp_cfg.get("cbf_k0", 1.0))
    base_cbf_k1 = float(low_qp_cfg.get("cbf_k1", 1.0))
    base_clf_k = float(low_qp_cfg.get("clf_k", 1.0))
    base_gamma_h = float(low_qp_cfg.get("hocbf_gamma_h", 1.0))
    base_gamma_hdot = float(low_qp_cfg.get("hocbf_gamma_hdot", 1.0))
    base_d_min_agent = float(safety_cfg.get("d_min_agent", 0.6))
    base_d_safe_obs = float(safety_cfg.get("d_safe_obs", 0.6))

    return {
        "action_dim": action_dim,
        "parameterize_cbf_constraints": parameterize_cbf_constraints,
        "w_clf": float(low_qp_cfg.get("w_clf", 10.0)),
        "w_cbf": float(low_qp_cfg.get("w_cbf", 100.0)),
        "cbf_slack_max": float(low_qp_cfg.get("cbf_slack_max", 1.0)),
        "cbf_k0": base_cbf_k0,
        "cbf_k1": base_cbf_k1,
        "clf_k": base_clf_k,
        "hocbf_gamma_h": base_gamma_h,
        "hocbf_gamma_hdot": base_gamma_hdot,
        "d_min_agent": base_d_min_agent,
        "d_safe_obs": base_d_safe_obs,
        "cbf_k0_min": float(low_qp_cfg.get("cbf_k0_min", 0.0)),
        "cbf_k0_max": float(low_qp_cfg.get("cbf_k0_max", max(2.0, base_cbf_k0 * 2.0))),
        "cbf_k1_min": float(low_qp_cfg.get("cbf_k1_min", 0.0)),
        "cbf_k1_max": float(low_qp_cfg.get("cbf_k1_max", max(2.0, base_cbf_k1 * 2.0))),
        "clf_k_min": float(low_qp_cfg.get("clf_k_min", 0.0)),
        "clf_k_max": float(low_qp_cfg.get("clf_k_max", max(2.0, base_clf_k * 2.0))),
        "hocbf_gamma_h_min": float(low_qp_cfg.get("hocbf_gamma_h_min", 0.0)),
        "hocbf_gamma_h_max": float(low_qp_cfg.get("hocbf_gamma_h_max", max(2.0, base_gamma_h * 2.0))),
        "hocbf_gamma_hdot_min": float(low_qp_cfg.get("hocbf_gamma_hdot_min", 0.0)),
        "hocbf_gamma_hdot_max": float(low_qp_cfg.get("hocbf_gamma_hdot_max", max(2.0, base_gamma_hdot * 2.0))),
        "d_min_agent_min": float(low_qp_cfg.get("d_min_agent_min", max(0.0, min(0.05, base_d_min_agent)))),
        "d_min_agent_max": float(low_qp_cfg.get("d_min_agent_max", base_d_min_agent)),
        "d_safe_obs_min": float(low_qp_cfg.get("d_safe_obs_min", max(0.0, min(0.05, base_d_safe_obs)))),
        "d_safe_obs_max": float(low_qp_cfg.get("d_safe_obs_max", base_d_safe_obs)),
        "f_residual_reference_enabled": bool(low_qp_cfg.get("f_residual_reference_enabled", True)),
        "f_ref_speed": float(low_qp_cfg.get("f_ref_speed", skills_cfg.get("ref_speed", 1.2))),
        "f_ref_kp": float(low_qp_cfg.get("f_ref_kp", 1.2)),
        "f_ref_slow_radius": float(low_qp_cfg.get("f_ref_slow_radius", skills_cfg.get("slow_radius", 1.5))),
        "f_ref_goal_stop_min_speed": float(
            low_qp_cfg.get("f_ref_goal_stop_min_speed", skills_cfg.get("goal_stop_min_speed", 0.0))
        ),
    }


def infer_low_phi_dim(cfg: Mapping[str, Any]) -> int:
    decoder_kwargs = build_low_decoder_kwargs(cfg)
    return LowQPDecoder.compute_phi_dim(
        int(decoder_kwargs["action_dim"]),
        parameterize_cbf_constraints=bool(decoder_kwargs["parameterize_cbf_constraints"]),
    )


def build_low_env(cfg: Mapping[str, Any], core: CoreEnv, *, torch_device: str) -> LowLevelOpenRLEnv:
    phi_dim = infer_low_phi_dim(cfg)
    n_skills = len(build_default_skill_library())
    low_cfg = LowLevelOpenRLEnvConfig(
        n_skills=n_skills,
        phi_dim=phi_dim,
        default_skill_id=int(cfg.get("teacher_pretrain", {}).get("default_skill_id", 2)),
        d_min_agent=float(cfg["safety"]["d_min_agent"]),
        d_safe_obs=float(cfg["safety"]["d_safe_obs"]),
        neighbor_perception_radius=float(cfg["env"].get("neighbor_radius", 0.0)),
        obstacle_perception_range=float(cfg["env"].get("lidar_range", 0.0)),
        include_local_obs_in_critic=bool(cfg.get("teacher_pretrain", {}).get("include_local_obs_in_critic", False)),
        reactivate_same_skill_on_switch=False,
        torch_device=torch_device,
        decoder_config=build_low_decoder_kwargs(cfg),
    )
    return LowLevelOpenRLEnv(core, low_cfg)


def build_low_agent(env: LowLevelOpenRLEnv, cfg: Mapping[str, Any], *, torch_device: str) -> LowDiffQPAgent:
    from hmarl_cbf.openrl_agents import LowDiffQPAgent

    n_skills = len(build_default_skill_library())
    model_cfg = low_openrl_cfg(n_skills=n_skills)
    hidden_dim = int(cfg.get("teacher_pretrain", {}).get("hidden_dim", 128))
    decoder_kwargs = build_low_decoder_kwargs(cfg)

    actor = LowQPActorNetwork(
        model_cfg,
        env.observation_space,
        env.action_space,
        device=torch_device,
        extra_args={"n_skills": n_skills, "hidden_dim": hidden_dim},
    )
    critic = LowQPCriticNetwork(
        model_cfg,
        env.state_space,
        device=torch_device,
        extra_args={"hidden_dim": hidden_dim},
    )
    decoder = LowQPDecoder(
        **decoder_kwargs,
    )
    bound_constraint_builder = getattr(env.action_adapter, "constraint_builder", None)
    bound_diff_qp_solver = getattr(env.action_adapter, "diff_qp_solver", None)
    agent = LowDiffQPAgent(actor, critic, decoder, torch_device=torch_device)
    if bound_constraint_builder is not None:
        agent.constraint_builder = bound_constraint_builder
    if bound_diff_qp_solver is not None:
        agent.diff_qp_solver = bound_diff_qp_solver
    agent.bind_env(env)
    return agent


def build_high_env(
    cfg: Mapping[str, Any],
    core: CoreEnv,
    *,
    low_level_executor=None,
) -> HighLevelOpenRLEnv:
    if low_level_executor is None:
        raise ValueError("build_high_env requires an explicit low_level_executor")
    sync_cfg = cfg.get("sync")
    if not isinstance(sync_cfg, Mapping):
        sync_cfg = cfg.get("synchronization", {})
    if not isinstance(sync_cfg, Mapping):
        sync_cfg = {}
    n_skills = len(build_default_skill_library())
    env_cfg = HighLevelOpenRLEnvConfig(
        n_skills=n_skills,
        coordinator_mode=str(sync_cfg.get("mode", "sync")),
        t_sync_max=int(sync_cfg.get("t_sync_max", 10)),
        gamma_high=float(cfg.get("train", {}).get("gamma_high", 0.99)),
        max_low_steps_per_high_step=int(cfg.get("openrl_train", {}).get("max_low_steps_per_high_step", 0)),
        freeze_on_reach=bool(cfg.get("openrl_train", {}).get("freeze_on_reach", True)),
        d_safe_obs=float(cfg["safety"]["d_safe_obs"]),
        high_option_progress_coef=float(cfg.get("train", {}).get("high_option_progress_coef", 0.0)),
        high_option_boundary_recovery_coef=float(cfg.get("train", {}).get("high_option_boundary_recovery_coef", 0.0)),
        high_option_boundary_threshold=float(cfg.get("train", {}).get("high_option_boundary_threshold", 0.0)),
        high_option_trap_relief_coef=float(cfg.get("train", {}).get("high_option_trap_relief_coef", 0.0)),
        high_option_trap_enter_coef=float(cfg.get("train", {}).get("high_option_trap_enter_coef", 0.0)),
        high_option_stuck_penalty_coef=float(cfg.get("train", {}).get("high_option_stuck_penalty_coef", 0.0)),
        high_option_stuck_blocked_threshold=float(cfg.get("train", {}).get("high_option_stuck_blocked_threshold", 0.5)),
        high_option_stuck_progress_threshold=float(cfg.get("train", {}).get("high_option_stuck_progress_threshold", 0.1)),
        high_option_stuck_speed_threshold=float(cfg.get("train", {}).get("high_option_stuck_speed_threshold", 0.2)),
        high_trap_blocked_lookahead=float(cfg.get("train", {}).get("high_trap_blocked_lookahead", 4.0)),
        high_trap_blocked_lateral_window=float(cfg.get("train", {}).get("high_trap_blocked_lateral_window", 3.0)),
        high_trap_blocked_extra_margin=float(cfg.get("train", {}).get("high_trap_blocked_extra_margin", 0.1)),
    )
    return HighLevelOpenRLEnv(core, env_cfg, low_level_executor=low_level_executor)


def build_high_net(env: HighLevelOpenRLEnv, *, torch_device: str) -> HighLevelMAPPONet:
    return HighLevelMAPPONet(env=env, device=torch_device, n_rollout_threads=1)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")


class CyclicSkillScheduler:
    def __init__(self, mode: str = "cyclic", seed: int = 0) -> None:
        self.mode = str(mode).strip().lower()
        self.rng = np.random.default_rng(int(seed))
        self.cursor: Dict[int, int] = {}

    def select(self, env: LowLevelOpenRLEnv, agent_id: int) -> int:
        available = env.get_available_skill_ids(int(agent_id))
        if not available:
            raise RuntimeError(f"agent {agent_id} has no available skill")
        if self.mode == "random":
            return int(self.rng.choice(np.asarray(available, dtype=np.int64)))
        idx = int(self.cursor.get(int(agent_id), 0)) % len(available)
        self.cursor[int(agent_id)] = idx + 1
        return int(available[idx])

    def assign_pending(self, env: LowLevelOpenRLEnv) -> None:
        if not env.has_pending_switch():
            return
        env.set_skill_map(
            {int(agent_id): self.select(env, int(agent_id)) for agent_id in env.pending_switch_agents},
            validate=True,
            require_all=False,
        )


class LowLevelPolicyExecutor:
    def __init__(self, agent: LowDiffQPAgent, n_skills: int) -> None:
        self.agent = agent
        self.n_skills = int(n_skills)
        self.device = agent.device
        self._recurrent_N = int(getattr(agent.actor, "_recurrent_N", 1))
        self._actor_hidden = int(getattr(agent.actor, "hidden_dim", 128))
        self._rnn_by_agent: Dict[int, torch.Tensor] = {}

    def reset(self, agent_ids: Sequence[int]) -> None:
        self._rnn_by_agent = {
            int(agent_id): torch.zeros((1, self._recurrent_N, self._actor_hidden), dtype=torch.float32, device=self.device)
            for agent_id in agent_ids
        }

    def __call__(self, states, obs_low, skill_targets, core_env):
        del core_env
        if self.agent._bound_adapter is None:
            raise RuntimeError("low-level executor requires a bound LowDiffQPAgent")
        agent_ids = sorted(int(agent_id) for agent_id in skill_targets.keys())
        if not self._rnn_by_agent:
            self.reset(agent_ids)

        obs_batch = []
        for agent_id in agent_ids:
            one_hot = np.zeros((self.n_skills,), dtype=np.float32)
            skill_id = int(skill_targets[agent_id]["skill_id"])
            if 0 <= skill_id < self.n_skills:
                one_hot[skill_id] = 1.0
            obs_batch.append(np.concatenate([np.asarray(obs_low[agent_id].flat, dtype=np.float32), one_hot], axis=0))
        obs_t = torch.as_tensor(np.stack(obs_batch, axis=0), dtype=torch.float32, device=self.device)
        rnn_t = torch.cat([self._rnn_by_agent[aid] for aid in agent_ids], dim=0)
        masks_t = torch.ones((len(agent_ids), 1), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = self.agent.actor.sample_phi(obs_t, rnn_t, masks_t, deterministic=True)
        next_rnn = out["rnn_states"]
        phi_np = np.asarray(out["phi"].detach().cpu().numpy(), dtype=np.float32)

        actions: Dict[int, np.ndarray] = {}
        for idx, agent_id in enumerate(agent_ids):
            self._rnn_by_agent[agent_id] = next_rnn[idx : idx + 1].detach().clone()
            neighbors = [states[other_id] for other_id in agent_ids if int(other_id) != int(agent_id)]
            solve_out = self.agent._bound_adapter.solve_numpy_for_agent(
                agent_id=int(agent_id),
                state_i=states[agent_id],
                neighbors=neighbors,
                obs_low=obs_low[agent_id],
                skill_id=int(skill_targets[agent_id]["skill_id"]),
                phi=phi_np[idx],
                safety_constraints=skill_targets[agent_id].get("safety_constraints", {}),
            )
            actions[int(agent_id)] = np.asarray(solve_out.action, dtype=np.float32).reshape(2).copy()
        return actions


class JointLowLevelPolicyExecutor:
    """Shared-rollout low-level executor used inside high-level training.

    This mirrors the original algorithm's structure:
    - the high-level environment owns switching/coordinator logic
    - the low-level policy runs every physical step
    - low-level rollout samples are recorded from the same shared rollout
    """

    def __init__(
        self,
        agent: LowDiffQPAgent,
        core_env: CoreEnv,
        *,
        n_skills: int,
        include_local_low_obs_in_critic: bool = False,
        deterministic: bool = False,
    ) -> None:
        self.agent = agent
        self.core_env = core_env
        self.n_skills = int(n_skills)
        self.include_local_low_obs_in_critic = bool(include_local_low_obs_in_critic)
        self.deterministic = bool(deterministic)
        self.device = agent.device
        self._current_skill_ids: Dict[int, int] = {}
        self._pending: Dict[int, Dict[str, Any]] = {}

    def reset(self, agent_ids: Sequence[int]) -> None:
        self.agent.reset()
        self._current_skill_ids.clear()
        self._pending.clear()
        if not self.agent._actor_rnn_by_agent:
            self.agent._init_agent_slots(agent_ids)

    def prepare_step_context(self, *, skill_ids: Mapping[int, int]) -> None:
        self._current_skill_ids = {int(agent_id): int(skill_id) for agent_id, skill_id in skill_ids.items()}

    def _build_infos(self, agent_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        critic_state = self.core_env.build_low_critic_obs(
            skill_ids=self._current_skill_ids,
            n_skills=self.n_skills,
            include_local_low_obs=bool(self.include_local_low_obs_in_critic),
        )
        infos: Dict[int, Dict[str, Any]] = {}
        for agent_id in agent_ids:
            infos[int(agent_id)] = {
                "critic_state": critic_state.copy(),
                "skill_id": int(self._current_skill_ids[int(agent_id)]),
            }
        return infos

    def __call__(
        self,
        states: Mapping[int, AgentState],
        obs_low: Mapping[int, AgentObsLow],
        skill_targets: Mapping[int, Dict[str, Any]],
        core_env: CoreEnv,
    ) -> tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        del core_env
        agent_ids = sorted(int(agent_id) for agent_id in skill_targets.keys())
        for agent_id in agent_ids:
            self._current_skill_ids[int(agent_id)] = int(skill_targets[int(agent_id)]["skill_id"])

        actor_obs = self.core_env.build_low_actor_obs(skill_ids=self._current_skill_ids, n_skills=self.n_skills)
        infos = self._build_infos(agent_ids)
        action_out = self.agent.act(
            {int(agent_id): actor_obs[int(agent_id)] for agent_id in agent_ids},
            infos,
            deterministic=self.deterministic,
        )

        executed_actions: Dict[int, np.ndarray] = {}
        adapter_info: Dict[int, Dict[str, Any]] = {}
        pending: Dict[int, Dict[str, Any]] = {}
        for agent_id in agent_ids:
            state_i = states[int(agent_id)]
            neighbors = [states[other_id] for other_id in agent_ids if int(other_id) != int(agent_id)]
            solve_out = self.agent._bound_adapter.solve_numpy_for_agent(
                agent_id=int(agent_id),
                state_i=state_i,
                neighbors=neighbors,
                obs_low=obs_low[int(agent_id)],
                skill_id=int(skill_targets[int(agent_id)]["skill_id"]),
                phi=np.asarray(action_out[int(agent_id)]["phi"], dtype=np.float32).reshape(-1),
                safety_constraints=skill_targets[int(agent_id)].get("safety_constraints", {}),
            )
            executed_actions[int(agent_id)] = np.asarray(solve_out.action, dtype=np.float32).reshape(2).copy()
            adapter_info[int(agent_id)] = {
                "adapter_mode": "torch_diff_qp",
                "phi": np.asarray(solve_out.phi, dtype=np.float32).copy(),
                "executed_action": np.asarray(solve_out.action, dtype=np.float32).reshape(2).copy(),
                "qp_feasible": bool(solve_out.qp_feasible),
                "qp_solver_status": str(solve_out.qp_solver_status),
                "qp_used_fallback": bool(solve_out.qp_used_fallback),
                "qp_H": np.asarray(solve_out.qp_param.H_mat, dtype=np.float32).reshape(2, 2).copy(),
                "qp_f": np.asarray(solve_out.qp_param.f_lin, dtype=np.float32).reshape(2).copy(),
                "qp_slack": np.asarray(solve_out.slack, dtype=np.float32).reshape(-1).copy(),
                "qp_cbf_slack": np.asarray(solve_out.cbf_slack, dtype=np.float32).reshape(-1).copy(),
                "qp_b_cbf": np.asarray(solve_out.b_cbf, dtype=np.float32).reshape(-1).copy(),
                "qp_b_clf": np.asarray(solve_out.b_clf, dtype=np.float32).reshape(-1).copy(),
                "neighbors_used": len(solve_out.neighbors_used),
                "obstacles_used": len(solve_out.obstacles_used),
                "perceived_neighbors": [
                    {
                        "agent_id": int(s.agent_id),
                        "position": np.asarray(s.position, dtype=np.float32).reshape(2).copy(),
                        "velocity": np.asarray(s.velocity, dtype=np.float32).reshape(2).copy(),
                        "goal": np.asarray(s.goal, dtype=np.float32).reshape(2).copy(),
                        "radius": float(s.radius),
                    }
                    for s in solve_out.neighbors_used
                ],
                "perceived_obstacles": [dict(obs) for obs in solve_out.obstacles_used],
                "safety_constraints": dict(solve_out.safety_constraints),
            }
            pending[int(agent_id)] = {
                "actor_obs": np.asarray(actor_obs[int(agent_id)], dtype=np.float32).reshape(-1).copy(),
                "critic_state": np.asarray(infos[int(agent_id)]["critic_state"], dtype=np.float32).reshape(-1).copy(),
                "obs_low": obs_low[int(agent_id)],
                "skill_id": int(skill_targets[int(agent_id)]["skill_id"]),
                "phi": np.asarray(action_out[int(agent_id)]["phi"], dtype=np.float32).reshape(-1).copy(),
                "logp": float(action_out[int(agent_id)]["logp"]),
                "value": float(action_out[int(agent_id)]["value"]),
                "actor_rnn_state": np.asarray(action_out[int(agent_id)]["actor_rnn_state"], dtype=np.float32).copy(),
                "critic_rnn_state": np.asarray(action_out[int(agent_id)]["critic_rnn_state"], dtype=np.float32).copy(),
            }
        self._pending = pending
        return executed_actions, adapter_info

    def record_transition(
        self,
        *,
        states: Mapping[int, AgentState],
        obs_low: Mapping[int, AgentObsLow],
        skill_targets: Mapping[int, Dict[str, Any]],
        executed_actions: Mapping[int, np.ndarray],
        step_res,
        next_states: Mapping[int, AgentState],
        next_obs_low: Mapping[int, AgentObsLow],
        skill_outputs: Mapping[int, Any],
        adapter_info: Mapping[int, Dict[str, Any]],
        option_k_by_agent: Mapping[int, int],
        done_agents: set[int],
    ) -> None:
        agent_ids = sorted(int(agent_id) for agent_id in states.keys())
        if not agent_ids:
            return
        unsafe_flags = dict(step_res.info.get("unsafe_flags", {}))
        reach_flags = dict(step_res.info.get("reach_flags", {}))
        next_critic_state = self.core_env.build_low_critic_obs(
            skill_ids=self._current_skill_ids,
            n_skills=self.n_skills,
            include_local_low_obs=bool(self.include_local_low_obs_in_critic),
        )
        next_masks = {
            int(agent_id): (
                torch.zeros((1, 1), dtype=torch.float32, device=self.device)
                if bool(
                    unsafe_flags.get(int(agent_id), False)
                    or reach_flags.get(int(agent_id), False)
                    or step_res.truncated
                    or int(agent_id) in done_agents
                )
                else torch.ones((1, 1), dtype=torch.float32, device=self.device)
            )
            for agent_id in agent_ids
        }
        next_critic_batch = np.stack(
            [np.asarray(next_critic_state, dtype=np.float32).reshape(-1) for _ in agent_ids],
            axis=0,
        )
        with torch.no_grad():
            next_values_t, _ = self.agent.critic.forward(
                torch.as_tensor(next_critic_batch, dtype=torch.float32, device=self.device),
                torch.cat([self.agent._critic_rnn_by_agent[aid] for aid in agent_ids], dim=0),
                torch.cat([next_masks[aid] for aid in agent_ids], dim=0),
            )

        for idx, agent_id in enumerate(agent_ids):
            pending = self._pending[int(agent_id)]
            skill_out = skill_outputs[int(agent_id)]
            step_info = {
                "skill_id": int(skill_out.skill_id),
                "switch_required": bool(skill_out.beta),
                "terminated_by_skill": bool(skill_out.beta),
                "intrinsic_reward": float(skill_out.intrinsic_reward),
                "reward_ext": float(step_res.rewards.get(int(agent_id), 0.0)),
                "reward_total": float(step_res.rewards.get(int(agent_id), 0.0)),
                "tau": int(skill_out.tau),
                "u_ref_skill": np.asarray(skill_out.u_ref_skill, dtype=np.float32).reshape(2).copy(),
                "safety_constraints": dict(skill_out.safety_constraints),
                "phi": np.asarray(pending["phi"], dtype=np.float32).copy(),
                "executed_action": np.asarray(executed_actions[int(agent_id)], dtype=np.float32).reshape(2).copy(),
                **dict(adapter_info.get(int(agent_id), {})),
            }
            self.agent.rollout.append(
                LowDiffQPRolloutSample(
                    step_index=int(self.agent._step_index),
                    agent_id=int(agent_id),
                    actor_obs=np.asarray(pending["actor_obs"], dtype=np.float32).copy(),
                    critic_state=np.asarray(pending["critic_state"], dtype=np.float32).copy(),
                    obs_low=pending["obs_low"],
                    skill_id=int(pending["skill_id"]),
                    phi=np.asarray(pending["phi"], dtype=np.float32).copy(),
                    logp=float(pending["logp"]),
                    value=float(pending["value"]),
                    reward=float(step_res.rewards.get(int(agent_id), 0.0)),
                    reward_int=float(skill_out.intrinsic_reward),
                    reward_ext=float(step_res.rewards.get(int(agent_id), 0.0)),
                    done=bool(
                        unsafe_flags.get(int(agent_id), False)
                        or reach_flags.get(int(agent_id), False)
                        or step_res.truncated
                        or int(agent_id) in done_agents
                    ),
                    option_k=int(option_k_by_agent.get(int(agent_id), 0)),
                    sync_switch=bool(step_info["switch_required"]),
                    bootstrap_value=float(next_values_t[idx].reshape(-1)[0].detach().cpu().item()),
                    actor_rnn_state=np.asarray(pending["actor_rnn_state"], dtype=np.float32).copy(),
                    critic_rnn_state=np.asarray(pending["critic_rnn_state"], dtype=np.float32).copy(),
                    info=step_info,
                )
            )
            self.agent._masks_by_agent[int(agent_id)] = next_masks[int(agent_id)]
        self.agent._step_index += 1
        self._pending.clear()
