from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from hmarl_cbf.openrl_models.high_mappo_net import HighLevelMAPPONet

if TYPE_CHECKING:
    from hmarl_cbf.openrl_train.teacher_dataset import HighLevelTeacherDatasetSample


@dataclass(slots=True)
class HighMAPPOAgentConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True


@dataclass(slots=True)
class HighMAPPOTeacherPretrainConfig:
    epochs: int = 5
    actor_lr: float | None = None
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5


@dataclass(slots=True)
class HighLevelRolloutSample:
    step_index: int
    agent_id: int
    actor_obs: np.ndarray
    critic_state: np.ndarray
    action_mask: np.ndarray
    option_k: int
    action: int
    logp: float
    value: float
    reward: float
    done: bool
    bootstrap_value: float
    actor_rnn_state: np.ndarray
    critic_rnn_state: np.ndarray
    advantage: float | None = None
    return_target: float | None = None
    info: Dict[str, Any] = field(default_factory=dict)


class HighMAPPOAgent:
    """High-level CTDE agent built around the OpenRL high-level net."""

    def __init__(
        self,
        net: HighLevelMAPPONet,
        config: HighMAPPOAgentConfig | None = None,
        *,
        actor_optimizer: Any | None = None,
        critic_optimizer: Any | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("HighMAPPOAgent requires PyTorch")
        self.net = net
        self.config = config or HighMAPPOAgentConfig()
        self.device = torch.device(self.net.device)

        self.actor = self.net.module.models["policy"]
        self.critic = self.net.module.models["critic"]

        self.actor_optimizer = actor_optimizer or torch.optim.Adam(
            self.actor.parameters(),
            lr=float(self.config.actor_lr),
        )
        self.critic_optimizer = critic_optimizer or torch.optim.Adam(
            self.critic.parameters(),
            lr=float(self.config.critic_lr),
        )

        self.rollout: List[HighLevelRolloutSample] = []
        self._actor_hidden = int(getattr(self.actor, "hidden_size", getattr(self.net.cfg, "hidden_size", 128)))
        self._critic_hidden = int(getattr(self.critic, "hidden_size", getattr(self.net.cfg, "hidden_size", 128)))
        self._recurrent_N = int(getattr(self.net.cfg, "recurrent_N", 1))
        self.reset()

    def _empty_rnn(self, batch: int, hidden: int) -> torch.Tensor:
        return torch.zeros((batch, self._recurrent_N, hidden), dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.rollout.clear()
        self._actor_rnn_by_agent = {
            int(agent_id): self._empty_rnn(1, self._actor_hidden)
            for agent_id in self.net.env.possible_agents
        }
        self._critic_rnn_by_agent = {
            int(agent_id): self._empty_rnn(1, self._critic_hidden)
            for agent_id in self.net.env.possible_agents
        }
        self._masks_by_agent = {
            int(agent_id): torch.ones((1, 1), dtype=torch.float32, device=self.device)
            for agent_id in self.net.env.possible_agents
        }
        self._step_index = 0

    @staticmethod
    def _sorted_agent_ids(obs: Mapping[int, np.ndarray]) -> List[int]:
        return sorted(int(agent_id) for agent_id in obs.keys())

    def act(
        self,
        obs: Mapping[int, np.ndarray],
        infos: Mapping[int, Mapping[str, Any]],
        valid_skill_ids: Mapping[int, Sequence[int]] | None = None,
        *,
        deterministic: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        agent_ids = self._sorted_agent_ids(obs)
        if not agent_ids:
            return {}
        obs_batch = np.stack([np.asarray(obs[aid], dtype=np.float32).reshape(-1) for aid in agent_ids], axis=0)
        critic_batch = np.stack(
            [np.asarray(infos[aid]["critic_state"], dtype=np.float32).reshape(-1) for aid in agent_ids],
            axis=0,
        )
        if valid_skill_ids is None:
            valid_action_ids = [list(infos[aid].get("valid_skill_ids", [])) for aid in agent_ids]
        else:
            valid_action_ids = [list(valid_skill_ids[int(aid)]) for aid in agent_ids]

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        critic_t = torch.as_tensor(critic_batch, dtype=torch.float32, device=self.device)
        actor_rnn = torch.cat([self._actor_rnn_by_agent[aid] for aid in agent_ids], dim=0)
        critic_rnn = torch.cat([self._critic_rnn_by_agent[aid] for aid in agent_ids], dim=0)
        masks = torch.cat([self._masks_by_agent[aid] for aid in agent_ids], dim=0)

        with torch.no_grad():
            actions_t, logp_t, next_actor_rnn = self.actor.forward_valid_actions(
                obs_t,
                actor_rnn,
                masks,
                valid_action_ids=valid_action_ids,
                deterministic=deterministic,
            )
            values_t, next_critic_rnn = self.critic.forward(critic_t, critic_rnn, masks)

        outputs: Dict[int, Dict[str, Any]] = {}
        for idx, agent_id in enumerate(agent_ids):
            self._actor_rnn_by_agent[agent_id] = next_actor_rnn[idx : idx + 1].detach().clone()
            self._critic_rnn_by_agent[agent_id] = next_critic_rnn[idx : idx + 1].detach().clone()
            mask = np.zeros((self.actor.action_dim,), dtype=np.float32)
            for skill_id in valid_action_ids[idx]:
                if 0 <= int(skill_id) < self.actor.action_dim:
                    mask[int(skill_id)] = 1.0
            outputs[agent_id] = {
                "action": int(actions_t[idx].reshape(-1)[0].detach().cpu().item()),
                "logp": float(logp_t[idx].reshape(-1)[0].detach().cpu().item()),
                "value": float(values_t[idx].reshape(-1)[0].detach().cpu().item()),
                "action_mask": mask,
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
        if hasattr(env, "decision_agent_ids"):
            decision_ids = [int(agent_id) for agent_id in env.decision_agent_ids()]
            decision_obs = {int(agent_id): obs[int(agent_id)] for agent_id in decision_ids}
            decision_infos = {int(agent_id): infos[int(agent_id)] for agent_id in decision_ids}
            env_valid_skill_ids = env.valid_skill_ids() if hasattr(env, "valid_skill_ids") else {
                int(agent_id): list(decision_infos[int(agent_id)].get("valid_skill_ids", []))
                for agent_id in decision_ids
            }
        else:
            decision_obs = obs
            decision_infos = infos
            env_valid_skill_ids = {
                int(agent_id): list(decision_infos[int(agent_id)].get("valid_skill_ids", []))
                for agent_id in decision_obs.keys()
            }
        action_out = self.act(
            decision_obs,
            decision_infos,
            valid_skill_ids=env_valid_skill_ids,
            deterministic=deterministic,
        )
        actions = {int(agent_id): int(item["action"]) for agent_id, item in action_out.items()}
        next_obs, rewards, terminations, truncations, next_infos = env.step(actions)

        decision_agent_ids = sorted(int(agent_id) for agent_id in action_out.keys())
        next_masks = {
            int(aid): torch.zeros((1, 1), dtype=torch.float32, device=self.device)
            if bool(terminations.get(aid, False) or truncations.get(aid, False))
            else torch.ones((1, 1), dtype=torch.float32, device=self.device)
            for aid in self._sorted_agent_ids(next_obs)
        }

        if decision_agent_ids:
            next_critic_batch = np.stack(
                [np.asarray(next_infos[aid]["critic_state"], dtype=np.float32).reshape(-1) for aid in decision_agent_ids],
                axis=0,
            )
            with torch.no_grad():
                next_values_t, _ = self.critic.forward(
                    torch.as_tensor(next_critic_batch, dtype=torch.float32, device=self.device),
                    torch.cat([self._critic_rnn_by_agent[aid] for aid in decision_agent_ids], dim=0),
                    torch.cat([next_masks[aid] for aid in decision_agent_ids], dim=0),
                )

            for idx, agent_id in enumerate(decision_agent_ids):
                self.rollout.append(
                    HighLevelRolloutSample(
                        step_index=int(self._step_index),
                        agent_id=int(agent_id),
                        actor_obs=np.asarray(obs[agent_id], dtype=np.float32).reshape(-1).copy(),
                        critic_state=np.asarray(infos[agent_id]["critic_state"], dtype=np.float32).reshape(-1).copy(),
                        action_mask=np.asarray(action_out[agent_id]["action_mask"], dtype=np.float32).reshape(-1).copy(),
                        option_k=int(infos[agent_id].get("option_k", 0)),
                        action=int(action_out[agent_id]["action"]),
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

        for agent_id, mask in next_masks.items():
            self._masks_by_agent[int(agent_id)] = mask
        self._step_index += 1
        return next_obs, rewards, terminations, truncations, next_infos, action_out

    def advantages_by_option(
        self,
        samples: Sequence[HighLevelRolloutSample] | None = None,
    ) -> Dict[tuple[int, int], float]:
        rollout = list(self.rollout if samples is None else samples)
        if not rollout:
            return {}
        self._compute_advantages(rollout)
        return {
            (int(sample.agent_id), int(sample.option_k)): float(sample.advantage if sample.advantage is not None else 0.0)
            for sample in rollout
        }

    def _compute_advantages(self, samples: Sequence[HighLevelRolloutSample]) -> None:
        by_agent: Dict[int, List[HighLevelRolloutSample]] = {}
        for sample in samples:
            by_agent.setdefault(int(sample.agent_id), []).append(sample)
        for agent_samples in by_agent.values():
            agent_samples.sort(key=lambda item: item.step_index)
            gae = 0.0
            for sample in reversed(agent_samples):
                nonterminal = 0.0 if sample.done else 1.0
                delta = sample.reward + self.config.gamma * sample.bootstrap_value * nonterminal - sample.value
                gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
                sample.advantage = float(gae)
                sample.return_target = float(gae + sample.value)

    def update(self, samples: Sequence[HighLevelRolloutSample] | None = None) -> Dict[str, float]:
        if torch is None:
            raise RuntimeError("HighMAPPOAgent requires PyTorch")
        rollout = list(self.rollout if samples is None else samples)
        if not rollout:
            return {
                "n_samples": 0.0,
                "loss_actor": 0.0,
                "loss_value": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
            }

        self._compute_advantages(rollout)
        advantages = np.asarray([float(sample.advantage) for sample in rollout], dtype=np.float32)
        if self.config.normalize_advantages and advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_obs = torch.as_tensor(
            np.stack([sample.actor_obs for sample in rollout], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        critic_obs = torch.as_tensor(
            np.stack([sample.critic_state for sample in rollout], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        action_masks = torch.as_tensor(
            np.stack([sample.action_mask for sample in rollout], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.asarray([sample.action for sample in rollout], dtype=np.int64).reshape(-1, 1),
            dtype=torch.long,
            device=self.device,
        )
        old_logp = torch.as_tensor(
            np.asarray([sample.logp for sample in rollout], dtype=np.float32).reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.as_tensor(
            np.asarray([sample.value for sample in rollout], dtype=np.float32).reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        returns = torch.as_tensor(
            np.asarray([sample.return_target for sample in rollout], dtype=np.float32).reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        advantages_t = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=self.device)
        actor_rnn = torch.as_tensor(
            np.stack([sample.actor_rnn_state for sample in rollout], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        critic_rnn = torch.as_tensor(
            np.stack([sample.critic_rnn_state for sample in rollout], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.ones((len(rollout), 1), dtype=torch.float32, device=self.device)

        loss_actor_all: List[float] = []
        loss_value_all: List[float] = []
        entropy_all: List[float] = []
        kl_all: List[float] = []
        clip_frac_all: List[float] = []

        n = len(rollout)
        batch_size = max(1, min(int(self.config.minibatch_size), n))
        for _ in range(max(1, int(self.config.ppo_epochs))):
            order = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx_np = order[start : start + batch_size]
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=self.device)

                new_logp, dist_entropy, _ = self.actor.eval_actions(
                    actor_obs.index_select(0, idx),
                    actor_rnn.index_select(0, idx),
                    actions.index_select(0, idx),
                    masks.index_select(0, idx),
                    action_masks=action_masks.index_select(0, idx),
                    active_masks=None,
                )
                new_values, _ = self.critic.forward(
                    critic_obs.index_select(0, idx),
                    critic_rnn.index_select(0, idx),
                    masks.index_select(0, idx),
                )
                new_logp = new_logp.reshape(-1)
                entropy_bonus = dist_entropy.reshape(-1).mean()
                new_values = new_values.reshape(-1)

                ratio = torch.exp(new_logp - old_logp.index_select(0, idx))
                adv_mb = advantages_t.index_select(0, idx)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * adv_mb
                actor_loss = -torch.minimum(surr1, surr2).mean()

                old_value_mb = old_values.index_select(0, idx)
                ret_mb = returns.index_select(0, idx)
                value_pred_clipped = old_value_mb + torch.clamp(
                    new_values - old_value_mb,
                    -self.config.clip_ratio,
                    self.config.clip_ratio,
                )
                value_loss_unclipped = (new_values - ret_mb) ** 2
                value_loss_clipped = (value_pred_clipped - ret_mb) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                (actor_loss - self.config.entropy_coef * entropy_bonus).backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.config.value_coef * value_loss).backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_logp.index_select(0, idx) - new_logp).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()
                loss_actor_all.append(float(actor_loss.detach().cpu().item()))
                loss_value_all.append(float(value_loss.detach().cpu().item()))
                entropy_all.append(float(entropy_bonus.detach().cpu().item()))
                kl_all.append(float(approx_kl.detach().cpu().item()))
                clip_frac_all.append(float(clip_frac.detach().cpu().item()))

        if samples is None:
            self.rollout.clear()
        return {
            "n_samples": float(n),
            "loss_actor": float(np.mean(loss_actor_all) if loss_actor_all else 0.0),
            "loss_value": float(np.mean(loss_value_all) if loss_value_all else 0.0),
            "entropy": float(np.mean(entropy_all) if entropy_all else 0.0),
            "approx_kl": float(np.mean(kl_all) if kl_all else 0.0),
            "clip_frac": float(np.mean(clip_frac_all) if clip_frac_all else 0.0),
        }

    def pretrain_from_teacher(
        self,
        samples: Sequence["HighLevelTeacherDatasetSample"],
        config: HighMAPPOTeacherPretrainConfig | None = None,
    ) -> Dict[str, float]:
        if torch is None:
            raise RuntimeError("HighMAPPOAgent requires PyTorch")
        pretrain_cfg = config or HighMAPPOTeacherPretrainConfig()
        teacher_samples = list(samples)
        if not teacher_samples:
            return {
                "n_samples": 0.0,
                "n_epochs": float(pretrain_cfg.epochs),
                "loss_action": 0.0,
                "entropy": 0.0,
                "action_acc": 0.0,
            }

        if pretrain_cfg.actor_lr is not None:
            for group in self.actor_optimizer.param_groups:
                group["lr"] = float(pretrain_cfg.actor_lr)

        actor_obs = torch.as_tensor(
            np.stack([sample.actor_obs for sample in teacher_samples], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        action_masks = torch.as_tensor(
            np.stack([sample.action_mask for sample in teacher_samples], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        teacher_actions = torch.as_tensor(
            np.asarray([sample.teacher_action for sample in teacher_samples], dtype=np.int64).reshape(-1),
            dtype=torch.long,
            device=self.device,
        )
        rnn_states = self._empty_rnn(len(teacher_samples), self._actor_hidden)
        masks = torch.ones((len(teacher_samples), 1), dtype=torch.float32, device=self.device)

        loss_all: List[float] = []
        entropy_all: List[float] = []
        acc_all: List[float] = []

        for epoch in range(max(1, int(pretrain_cfg.epochs))):
            actor_features, _ = self.actor._forward_features(actor_obs, rnn_states, masks)
            logits = self.actor._masked_logits(actor_features, action_masks)
            dist = torch.distributions.Categorical(logits=logits)
            loss_action = -dist.log_prob(teacher_actions).mean()
            entropy = dist.entropy().mean()
            loss = loss_action - float(pretrain_cfg.entropy_coef) * entropy

            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(pretrain_cfg.max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(pretrain_cfg.max_grad_norm))
            self.actor_optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                acc = (pred == teacher_actions).float().mean()
            print("teacher_high_pretrain_epoch", int(epoch + 1))
            print("teacher_high_pretrain_loss_action", float(loss_action.detach().cpu().item()))
            print("teacher_high_pretrain_entropy", float(entropy.detach().cpu().item()))
            print("teacher_high_pretrain_action_acc", float(acc.detach().cpu().item()))
            loss_all.append(float(loss_action.detach().cpu().item()))
            entropy_all.append(float(entropy.detach().cpu().item()))
            acc_all.append(float(acc.detach().cpu().item()))

        return {
            "n_samples": float(len(teacher_samples)),
            "n_epochs": float(pretrain_cfg.epochs),
            "loss_action": float(np.mean(loss_all) if loss_all else 0.0),
            "entropy": float(np.mean(entropy_all) if entropy_all else 0.0),
            "action_acc": float(np.mean(acc_all) if acc_all else 0.0),
        }

    def save(self, path: str | Path) -> None:
        if torch is None:
            raise RuntimeError("HighMAPPOAgent requires PyTorch")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "config": asdict(self.config),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        if torch is None:
            raise RuntimeError("HighMAPPOAgent requires PyTorch")
        payload = torch.load(Path(path), map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
