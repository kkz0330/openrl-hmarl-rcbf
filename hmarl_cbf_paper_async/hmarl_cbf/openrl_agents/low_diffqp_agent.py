from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from hmarl_cbf.control import ConstraintBuilder, TorchDifferentiableQPSolver
from hmarl_cbf.openrl_agents.low_diffqp_executor import TorchDiffQPActionAdapter
from hmarl_cbf.openrl_models.low_qp_actor import LowQPActorNetwork
from hmarl_cbf.openrl_models.low_qp_critic import LowQPCriticNetwork
from hmarl_cbf.openrl_models.low_qp_decoder import LowQPDecoder
from hmarl_cbf.types import AgentObsLow, AgentState, QPParam


@dataclass(slots=True)
class LowDiffQPAgentConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    ppo_epochs: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    slack_coef: float = 0.02
    cbf_slack_coef: float = 0.05
    ext_reward_coef: float = 0.0
    reward_mix_eta: float = 0.0
    divide_high_adv_by_n_agents: bool = True
    reset_on_sync_switch: bool = True
    safety_margin_coef: float = 0.0
    safety_margin_h_agent: float = 0.0
    safety_margin_h_obstacle: float = 0.0


@dataclass(slots=True)
class LowDiffQPTeacherPretrainConfig:
    epochs: int = 5
    shuffle: bool = True
    action_coef: float = 1.0
    slack_coef: float = 0.02
    cbf_slack_coef: float = 0.05
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5


@dataclass(slots=True)
class LowDiffQPRolloutSample:
    step_index: int
    agent_id: int
    actor_obs: np.ndarray
    critic_state: np.ndarray
    obs_low: AgentObsLow
    skill_id: int
    phi: np.ndarray
    logp: float
    value: float
    reward: float
    reward_int: float
    reward_ext: float
    done: bool
    option_k: int
    sync_switch: bool
    bootstrap_value: float
    actor_rnn_state: np.ndarray
    critic_rnn_state: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)
    advantage: float | None = None
    return_target: float | None = None


class LowDiffQPAgent:
    """Low-level PPO agent with torch diff-QP in the update loop."""

    def __init__(
        self,
        actor: LowQPActorNetwork,
        critic: LowQPCriticNetwork,
        decoder: LowQPDecoder,
        *,
        config: LowDiffQPAgentConfig | None = None,
        diff_qp_solver: TorchDifferentiableQPSolver | None = None,
        constraint_builder: ConstraintBuilder | None = None,
        actor_optimizer: Any | None = None,
        critic_optimizer: Any | None = None,
        torch_device: str | torch.device = "cpu",
    ) -> None:
        if torch is None:
            raise RuntimeError("LowDiffQPAgent requires PyTorch")
        self.actor = actor
        self.critic = critic
        self.decoder = decoder
        self.config = config or LowDiffQPAgentConfig()
        self.device = torch.device(torch_device)

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.decoder.to(self.device)

        self.diff_qp_solver = diff_qp_solver or TorchDifferentiableQPSolver()
        self.constraint_builder = constraint_builder or ConstraintBuilder()

        self.actor_optimizer = actor_optimizer or torch.optim.Adam(
            list(self.actor.parameters()) + list(self.decoder.parameters()),
            lr=float(self.config.actor_lr),
        )
        self.critic_optimizer = critic_optimizer or torch.optim.Adam(
            self.critic.parameters(),
            lr=float(self.config.critic_lr),
        )

        self.rollout: List[LowDiffQPRolloutSample] = []
        self._actor_hidden = int(getattr(self.actor, "hidden_dim", getattr(self.actor, "hidden_size", 128)))
        self._critic_hidden = int(getattr(self.critic, "hidden_dim", getattr(self.critic, "hidden_size", 128)))
        self._recurrent_N = int(getattr(self.actor, "_recurrent_N", 1))
        self._bound_env = None
        self._bound_adapter: TorchDiffQPActionAdapter | None = None
        self.reset()

    def _empty_rnn(self, batch: int, hidden: int) -> torch.Tensor:
        return torch.zeros((batch, self._recurrent_N, hidden), dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.rollout.clear()
        self._actor_rnn_by_agent: Dict[int, torch.Tensor] = {}
        self._critic_rnn_by_agent: Dict[int, torch.Tensor] = {}
        self._masks_by_agent: Dict[int, torch.Tensor] = {}
        self._step_index = 0
        if self._bound_env is not None:
            self._init_agent_slots(self._bound_env.possible_agents)

    def _init_agent_slots(self, agent_ids: Sequence[int]) -> None:
        self._actor_rnn_by_agent = {
            int(agent_id): self._empty_rnn(1, self._actor_hidden)
            for agent_id in agent_ids
        }
        self._critic_rnn_by_agent = {
            int(agent_id): self._empty_rnn(1, self._critic_hidden)
            for agent_id in agent_ids
        }
        self._masks_by_agent = {
            int(agent_id): torch.ones((1, 1), dtype=torch.float32, device=self.device)
            for agent_id in agent_ids
        }

    def bind_env(self, env) -> None:
        self._bound_env = env
        self._init_agent_slots(env.possible_agents)
        self._bound_adapter = TorchDiffQPActionAdapter(
            decoder=self.decoder,
            diff_qp_solver=self.diff_qp_solver,
            constraint_builder=self.constraint_builder,
            neighbor_perception_radius=env.config.neighbor_perception_radius,
            obstacle_perception_range=env.config.obstacle_perception_range,
            torch_device=str(self.device),
        )
        env.action_adapter = self._bound_adapter

    @staticmethod
    def _sorted_agent_ids(obs: Mapping[int, np.ndarray]) -> List[int]:
        return sorted(int(agent_id) for agent_id in obs.keys())

    def act(
        self,
        obs: Mapping[int, np.ndarray],
        infos: Mapping[int, Mapping[str, Any]],
        *,
        deterministic: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        agent_ids = self._sorted_agent_ids(obs)
        if not self._actor_rnn_by_agent:
            self._init_agent_slots(agent_ids)

        obs_batch = np.stack([np.asarray(obs[aid], dtype=np.float32).reshape(-1) for aid in agent_ids], axis=0)
        critic_batch = np.stack(
            [np.asarray(infos[aid]["critic_state"], dtype=np.float32).reshape(-1) for aid in agent_ids],
            axis=0,
        )

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        critic_t = torch.as_tensor(critic_batch, dtype=torch.float32, device=self.device)
        actor_rnn = torch.cat([self._actor_rnn_by_agent[aid] for aid in agent_ids], dim=0)
        critic_rnn = torch.cat([self._critic_rnn_by_agent[aid] for aid in agent_ids], dim=0)
        masks = torch.cat([self._masks_by_agent[aid] for aid in agent_ids], dim=0)

        with torch.no_grad():
            sample = self.actor.sample_phi(obs_t, actor_rnn, masks, deterministic=deterministic)
            values_t, next_critic_rnn = self.critic.forward(critic_t, critic_rnn, masks)

        next_actor_rnn = sample["rnn_states"]
        phi_t = sample["phi"]
        logp_t = sample["logp"]

        outputs: Dict[int, Dict[str, Any]] = {}
        for idx, agent_id in enumerate(agent_ids):
            self._actor_rnn_by_agent[agent_id] = next_actor_rnn[idx : idx + 1].detach().clone()
            self._critic_rnn_by_agent[agent_id] = next_critic_rnn[idx : idx + 1].detach().clone()
            outputs[agent_id] = {
                "phi": np.asarray(phi_t[idx].detach().cpu().numpy(), dtype=np.float32).reshape(-1).copy(),
                "logp": float(logp_t[idx].reshape(-1)[0].detach().cpu().item()),
                "value": float(values_t[idx].reshape(-1)[0].detach().cpu().item()),
                "actor_rnn_state": actor_rnn[idx].detach().cpu().numpy().astype(np.float32).copy(),
                "critic_rnn_state": critic_rnn[idx].detach().cpu().numpy().astype(np.float32).copy(),
            }
        return outputs

    def step_env(
        self,
        env,
        obs: Mapping[int, np.ndarray],
        infos: Mapping[int, Mapping[str, Any]],
        *,
        deterministic: bool = False,
    ):
        if self._bound_env is not env:
            self.bind_env(env)

        current_obs_low = env.core_env.get_obs_low()
        action_out = self.act(obs, infos, deterministic=deterministic)
        phi_actions = {int(agent_id): np.asarray(item["phi"], dtype=np.float32).reshape(-1) for agent_id, item in action_out.items()}
        next_obs, rewards, terminations, truncations, next_infos = env.step(phi_actions)

        agent_ids = self._sorted_agent_ids(obs)
        next_critic_batch = np.stack(
            [np.asarray(next_infos[aid]["critic_state"], dtype=np.float32).reshape(-1) for aid in agent_ids],
            axis=0,
        )
        next_masks = {
            int(aid): torch.zeros((1, 1), dtype=torch.float32, device=self.device)
            if bool(terminations.get(aid, False) or truncations.get(aid, False))
            else torch.ones((1, 1), dtype=torch.float32, device=self.device)
            for aid in agent_ids
        }
        with torch.no_grad():
            next_values_t, _ = self.critic.forward(
                torch.as_tensor(next_critic_batch, dtype=torch.float32, device=self.device),
                torch.cat([self._critic_rnn_by_agent[aid] for aid in agent_ids], dim=0),
                torch.cat([next_masks[aid] for aid in agent_ids], dim=0),
            )

        for idx, agent_id in enumerate(agent_ids):
            self.rollout.append(
                LowDiffQPRolloutSample(
                    step_index=int(self._step_index),
                    agent_id=int(agent_id),
                    actor_obs=np.asarray(obs[agent_id], dtype=np.float32).reshape(-1).copy(),
                    critic_state=np.asarray(infos[agent_id]["critic_state"], dtype=np.float32).reshape(-1).copy(),
                    obs_low=current_obs_low[agent_id],
                    skill_id=int(infos[agent_id]["skill_id"]),
                    phi=np.asarray(action_out[agent_id]["phi"], dtype=np.float32).reshape(-1).copy(),
                    logp=float(action_out[agent_id]["logp"]),
                    value=float(action_out[agent_id]["value"]),
                    reward=float(rewards[agent_id]),
                    done=bool(terminations.get(agent_id, False) or truncations.get(agent_id, False)),
                    bootstrap_value=float(next_values_t[idx].reshape(-1)[0].detach().cpu().item()),
                    actor_rnn_state=np.asarray(action_out[agent_id]["actor_rnn_state"], dtype=np.float32).copy(),
                    critic_rnn_state=np.asarray(action_out[agent_id]["critic_rnn_state"], dtype=np.float32).copy(),
                    info=dict(next_infos.get(agent_id, {})),
                )
            )
            self._masks_by_agent[agent_id] = next_masks[agent_id]

        self._step_index += 1
        return next_obs, rewards, terminations, truncations, next_infos, action_out

    def _solve_teacher_sample(
        self,
        sample: Any,
        *,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, QPParam, Any, torch.Tensor]:
        if self._bound_adapter is None:
            raise RuntimeError("LowDiffQPAgent must be bound to a LowLevelOpenRLEnv before teacher pretraining")

        actor_obs_t = torch.as_tensor(
            np.asarray(sample.actor_obs, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32,
            device=self.device,
        )
        masks_t = torch.ones((1, 1), dtype=torch.float32, device=self.device)
        actor_rnn_t = self._empty_rnn(1, self._actor_hidden)
        sample_out = self.actor.sample_phi(actor_obs_t, actor_rnn_t, masks_t, deterministic=deterministic)
        phi_t = sample_out["phi"]
        solve_action, qp_param, qp_out, _, _ = self._bound_adapter.solve_torch_for_agent(
            agent_id=int(sample.agent_id),
            state_i=sample.state,
            neighbors=list(sample.neighbors),
            obs_low=sample.obs_low,
            skill_id=int(sample.skill_id),
            phi=phi_t,
            safety_constraints=dict(sample.safety_constraints),
        )
        return solve_action, qp_param, qp_out, sample_out["entropy"].reshape(())

    def _extract_local_context(
        self,
        sample: LowDiffQPRolloutSample,
    ) -> tuple[AgentState, List[AgentState], Dict[str, Any]]:
        pos = np.asarray(sample.obs_low.self_state[:2], dtype=np.float32).reshape(2)
        vel = np.asarray(sample.obs_low.self_state[2:4], dtype=np.float32).reshape(2)
        goal = pos + np.asarray(sample.obs_low.goal_relative, dtype=np.float32).reshape(2)
        state = AgentState(
            agent_id=int(sample.agent_id),
            position=pos,
            velocity=vel,
            goal=goal,
        )
        neighbors: List[AgentState] = []
        for item in list(sample.info.get("perceived_neighbors", [])):
            neighbors.append(
                AgentState(
                    agent_id=int(item["agent_id"]),
                    position=np.asarray(item["position"], dtype=np.float32).reshape(2),
                    velocity=np.asarray(item["velocity"], dtype=np.float32).reshape(2),
                    goal=np.asarray(item["goal"], dtype=np.float32).reshape(2),
                    radius=float(item.get("radius", 0.2)),
                )
            )
        safety_constraints = dict(sample.info.get("safety_constraints", {}))
        return state, neighbors, safety_constraints

    def _compute_advantages(
        self,
        samples: Sequence[LowDiffQPRolloutSample],
        *,
        high_adv_by_option: Mapping[tuple[int, int], float] | None = None,
        n_agents: int = 1,
    ) -> None:
        by_agent: Dict[int, List[LowDiffQPRolloutSample]] = {}
        for sample in samples:
            by_agent.setdefault(int(sample.agent_id), []).append(sample)
        high_adv_by_option = dict(high_adv_by_option or {})
        eta = float(max(0.0, min(1.0, self.config.reward_mix_eta)))
        denom = float(max(1, int(n_agents))) if self.config.divide_high_adv_by_n_agents else 1.0
        margin_coef = float(max(0.0, self.config.safety_margin_coef))
        h_agent_margin = float(max(0.0, self.config.safety_margin_h_agent))
        h_obs_margin = float(max(0.0, self.config.safety_margin_h_obstacle))
        for agent_samples in by_agent.values():
            agent_samples.sort(key=lambda item: item.step_index)
            running = 0.0
            for sample in reversed(agent_samples):
                if sample.done or (self.config.reset_on_sync_switch and sample.sync_switch):
                    running = 0.0
                high_adv = float(high_adv_by_option.get((int(sample.agent_id), int(sample.option_k)), 0.0))
                high_term = high_adv / denom
                min_h_agent = float(sample.info.get("min_h_agent", 0.0))
                min_h_obstacle = float(sample.info.get("min_h_obstacle", 0.0))
                safety_penalty = 0.0
                if margin_coef > 0.0:
                    if h_agent_margin > 0.0 and min_h_agent < h_agent_margin:
                        safety_penalty += (h_agent_margin - min_h_agent) ** 2
                    if h_obs_margin > 0.0 and min_h_obstacle < h_obs_margin:
                        safety_penalty += (h_obs_margin - min_h_obstacle) ** 2
                    safety_penalty *= margin_coef
                mixed_reward = (
                    eta * high_term
                    + (1.0 - eta) * float(sample.reward_int)
                    + float(self.config.ext_reward_coef) * float(sample.reward_ext)
                    - float(safety_penalty)
                )
                running = float(mixed_reward) + float(self.config.gamma) * running
                sample.info["reward_mix"] = float(mixed_reward)
                sample.info["high_adv_mix"] = float(high_term)
                sample.info["safety_margin_penalty"] = float(safety_penalty)
                sample.advantage = float(running - sample.value)
                sample.return_target = float(running)

    def update(
        self,
        samples: Sequence[LowDiffQPRolloutSample] | None = None,
        *,
        high_adv_by_option: Mapping[tuple[int, int], float] | None = None,
        n_agents: int = 1,
    ) -> Dict[str, float]:
        if torch is None:
            raise RuntimeError("LowDiffQPAgent requires PyTorch")
        rollout = list(self.rollout if samples is None else samples)
        if not rollout:
            return {
                "n_updates": 0.0,
                "loss_actor": 0.0,
                "loss_value": 0.0,
                "entropy": 0.0,
                "loss_slack": 0.0,
                "low_f_mean_x": 0.0,
                "low_f_mean_y": 0.0,
                "low_h_eig_min": 0.0,
                "low_h_eig_max": 0.0,
            }
        if self._bound_adapter is None:
            raise RuntimeError("LowDiffQPAgent must be bound to a LowLevelOpenRLEnv before update")

        self._compute_advantages(
            rollout,
            high_adv_by_option=high_adv_by_option,
            n_agents=int(n_agents),
        )
        adv_np = np.asarray([float(sample.advantage) for sample in rollout], dtype=np.float32)
        if self.config.normalize_advantages and adv_np.size > 1:
            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        loss_actor_all: List[float] = []
        loss_value_all: List[float] = []
        entropy_all: List[float] = []
        slack_all: List[float] = []
        f_x_vals: List[float] = []
        f_y_vals: List[float] = []
        h_eig_min_vals: List[float] = []
        h_eig_max_vals: List[float] = []
        n_updates = 0

        n_epochs = max(1, int(self.config.ppo_epochs))
        for _ in range(n_epochs):
            order = np.random.permutation(len(rollout))
            for raw_idx in order:
                sample = rollout[int(raw_idx)]
                state, neighbors, safety_constraints = self._extract_local_context(sample)

                actor_obs_t = torch.as_tensor(sample.actor_obs.reshape(1, -1), dtype=torch.float32, device=self.device)
                critic_obs_t = torch.as_tensor(sample.critic_state.reshape(1, -1), dtype=torch.float32, device=self.device)
                phi_t = torch.as_tensor(sample.phi.reshape(1, -1), dtype=torch.float32, device=self.device)
                actor_rnn_t = torch.as_tensor(sample.actor_rnn_state.reshape(1, self._recurrent_N, self._actor_hidden), dtype=torch.float32, device=self.device)
                critic_rnn_t = torch.as_tensor(sample.critic_rnn_state.reshape(1, self._recurrent_N, self._critic_hidden), dtype=torch.float32, device=self.device)
                masks_t = torch.ones((1, 1), dtype=torch.float32, device=self.device)

                new_logp, dist_entropy, _ = self.actor.eval_actions(
                    actor_obs_t,
                    actor_rnn_t,
                    phi_t,
                    masks_t,
                    action_masks=None,
                    active_masks=None,
                )
                value_pred, _ = self.critic.forward(critic_obs_t, critic_rnn_t, masks_t)
                solve_action, qp_param, qp_out, _, _ = self._bound_adapter.solve_torch_for_agent(
                    agent_id=int(sample.agent_id),
                    state_i=state,
                    neighbors=neighbors,
                    obs_low=sample.obs_low,
                    skill_id=int(sample.skill_id),
                    phi=phi_t,
                    safety_constraints=safety_constraints,
                )
                del solve_action

                old_logp = torch.as_tensor(float(sample.logp), dtype=torch.float32, device=self.device)
                adv_t = torch.as_tensor(float(adv_np[int(raw_idx)]), dtype=torch.float32, device=self.device)
                ratio = torch.exp(new_logp.reshape(-1)[0] - old_logp)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * adv_t
                actor_loss = -torch.minimum(surr1, surr2)

                ret_t = torch.as_tensor(float(sample.return_target), dtype=torch.float32, device=self.device)
                value_loss = 0.5 * (value_pred.reshape(-1)[0] - ret_t) ** 2

                entropy_bonus = dist_entropy.reshape(-1)[0]
                slack_loss = self.config.slack_coef * torch.sum(qp_out.slack ** 2)
                if qp_out.cbf_slack.numel() > 0:
                    slack_loss = slack_loss + self.config.cbf_slack_coef * torch.mean(qp_out.cbf_slack ** 2)

                self.actor_optimizer.zero_grad(set_to_none=True)
                (actor_loss - self.config.entropy_coef * entropy_bonus + slack_loss).backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.decoder.parameters()),
                        self.config.max_grad_norm,
                    )
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.config.value_coef * value_loss).backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                f_vec = qp_param.f_lin.reshape(-1)
                h_eigs = torch.linalg.eigvalsh(qp_param.H_mat.reshape(2, 2))
                loss_actor_all.append(float(actor_loss.detach().cpu().item()))
                loss_value_all.append(float(value_loss.detach().cpu().item()))
                entropy_all.append(float(entropy_bonus.detach().cpu().item()))
                slack_all.append(float(slack_loss.detach().cpu().item()))
                f_x_vals.append(float(f_vec[0].detach().cpu().item()))
                f_y_vals.append(float(f_vec[1].detach().cpu().item()))
                h_eig_min_vals.append(float(h_eigs[0].detach().cpu().item()))
                h_eig_max_vals.append(float(h_eigs[-1].detach().cpu().item()))
                n_updates += 1

        if samples is None:
            self.rollout.clear()
        return {
            "n_updates": float(n_updates),
            "loss_actor": float(np.mean(loss_actor_all) if loss_actor_all else 0.0),
            "loss_value": float(np.mean(loss_value_all) if loss_value_all else 0.0),
            "entropy": float(np.mean(entropy_all) if entropy_all else 0.0),
            "loss_slack": float(np.mean(slack_all) if slack_all else 0.0),
            "low_f_mean_x": float(np.mean(f_x_vals) if f_x_vals else 0.0),
            "low_f_mean_y": float(np.mean(f_y_vals) if f_y_vals else 0.0),
            "low_h_eig_min": float(np.mean(h_eig_min_vals) if h_eig_min_vals else 0.0),
            "low_h_eig_max": float(np.mean(h_eig_max_vals) if h_eig_max_vals else 0.0),
        }

    def pretrain_from_teacher(
        self,
        samples: Sequence[Any],
        *,
        config: LowDiffQPTeacherPretrainConfig | None = None,
    ) -> Dict[str, float]:
        if torch is None:
            raise RuntimeError("LowDiffQPAgent requires PyTorch")
        if self._bound_adapter is None:
            raise RuntimeError("LowDiffQPAgent must be bound to a LowLevelOpenRLEnv before teacher pretraining")
        cfg = config or LowDiffQPTeacherPretrainConfig()
        teacher_samples = list(samples)
        if not teacher_samples:
            return {
                "n_updates": 0.0,
                "loss_action": 0.0,
                "loss_slack": 0.0,
                "entropy": 0.0,
                "teacher_action_mae": 0.0,
                "teacher_feasible_rate": 0.0,
            }

        loss_action_all: List[float] = []
        loss_slack_all: List[float] = []
        entropy_all: List[float] = []
        mae_all: List[float] = []
        feasible_all: List[float] = []
        n_updates = 0

        n_epochs = max(1, int(cfg.epochs))
        for epoch_idx in range(n_epochs):
            epoch_action_losses: List[float] = []
            epoch_slack_losses: List[float] = []
            epoch_mae: List[float] = []
            order = np.arange(len(teacher_samples), dtype=np.int64)
            if cfg.shuffle and order.size > 1:
                order = np.random.permutation(order)
            for idx in order.tolist():
                sample = teacher_samples[int(idx)]
                solve_action, _, qp_out, entropy_bonus = self._solve_teacher_sample(sample, deterministic=True)
                target_t = torch.as_tensor(
                    np.asarray(sample.teacher_action, dtype=np.float32).reshape(2),
                    dtype=torch.float32,
                    device=self.device,
                )
                action_loss = float(cfg.action_coef) * 0.5 * torch.sum((solve_action.reshape(2) - target_t) ** 2)
                slack_loss = float(cfg.slack_coef) * torch.sum(qp_out.slack ** 2)
                if qp_out.cbf_slack.numel() > 0:
                    slack_loss = slack_loss + float(cfg.cbf_slack_coef) * torch.mean(qp_out.cbf_slack ** 2)

                total_loss = action_loss + slack_loss - float(cfg.entropy_coef) * entropy_bonus
                self.actor_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if cfg.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.decoder.parameters()),
                        float(cfg.max_grad_norm),
                    )
                self.actor_optimizer.step()

                diff = torch.abs(solve_action.reshape(2) - target_t)
                loss_action_all.append(float(action_loss.detach().cpu().item()))
                loss_slack_all.append(float(slack_loss.detach().cpu().item()))
                entropy_all.append(float(entropy_bonus.detach().cpu().item()))
                mae_all.append(float(torch.mean(diff).detach().cpu().item()))
                feasible_all.append(1.0 if bool(getattr(sample, "teacher_qp_feasible", True)) else 0.0)
                epoch_action_losses.append(float(action_loss.detach().cpu().item()))
                epoch_slack_losses.append(float(slack_loss.detach().cpu().item()))
                epoch_mae.append(float(torch.mean(diff).detach().cpu().item()))
                n_updates += 1

            print("teacher_pretrain_epoch", int(epoch_idx + 1))
            print("teacher_pretrain_loss_action", float(np.mean(epoch_action_losses) if epoch_action_losses else 0.0))
            print("teacher_pretrain_loss_slack", float(np.mean(epoch_slack_losses) if epoch_slack_losses else 0.0))
            print("teacher_pretrain_action_mae", float(np.mean(epoch_mae) if epoch_mae else 0.0))

        return {
            "n_updates": float(n_updates),
            "loss_action": float(np.mean(loss_action_all) if loss_action_all else 0.0),
            "loss_slack": float(np.mean(loss_slack_all) if loss_slack_all else 0.0),
            "entropy": float(np.mean(entropy_all) if entropy_all else 0.0),
            "teacher_action_mae": float(np.mean(mae_all) if mae_all else 0.0),
            "teacher_feasible_rate": float(np.mean(feasible_all) if feasible_all else 0.0),
        }

    def save(self, path: str | Path) -> None:
        if torch is None:
            raise RuntimeError("LowDiffQPAgent requires PyTorch")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "decoder": self.decoder.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "config": asdict(self.config),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        if torch is None:
            raise RuntimeError("LowDiffQPAgent requires PyTorch")
        payload = torch.load(Path(path), map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.decoder.load_state_dict(payload["decoder"])
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
