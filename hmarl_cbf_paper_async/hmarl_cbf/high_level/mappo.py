from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional backend
    torch = None  # type: ignore[assignment]

from hmarl_cbf.types import AgentObsHigh, HighOptionTransition


@dataclass(slots=True)
class MAPPOConfig:
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 64
    normalize_advantages: bool = True


class OnPolicyMAPPO:
    """Option-level MAPPO updater for high-level discrete skill policy."""

    def __init__(self, policy: torch.nn.Module, optimizer: torch.optim.Optimizer, config: MAPPOConfig | None = None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for MAPPO updater")
        self.policy = policy
        self.optimizer = optimizer
        self.config = config or MAPPOConfig()

    @staticmethod
    def _obs_to_vec(obs: AgentObsHigh) -> np.ndarray:
        return np.concatenate([obs.self_state, obs.goal_relative, obs.neighbor_summary], axis=0).astype(np.float32)

    def _build_batch(self, transitions: Iterable[HighOptionTransition]) -> Dict[str, torch.Tensor]:
        data = list(transitions)
        if len(data) == 0:
            raise ValueError("high-level transition list is empty")

        # Keep deterministic ordering for reproducibility.
        data.sort(key=lambda x: (x.k, x.agent_id, x.t_start))

        obs = np.stack([self._obs_to_vec(item.obs_high) for item in data], axis=0).astype(np.float32)
        z = np.asarray([item.skill_id for item in data], dtype=np.int64)
        old_logp = np.asarray([item.logp for item in data], dtype=np.float32)
        old_value = np.asarray([item.value for item in data], dtype=np.float32)
        returns = np.asarray(
            [item.value_target if item.value_target is not None else item.return_ext for item in data],
            dtype=np.float32,
        )
        adv = np.asarray(
            [item.advantage if item.advantage is not None else (ret - val) for item, ret, val in zip(data, returns, old_value)],
            dtype=np.float32,
        )
        if self.config.normalize_advantages and len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return {
            "obs": torch.as_tensor(obs, dtype=torch.float32),
            "z": torch.as_tensor(z, dtype=torch.long),
            "old_logp": torch.as_tensor(old_logp, dtype=torch.float32),
            "old_value": torch.as_tensor(old_value, dtype=torch.float32),
            "returns": torch.as_tensor(returns, dtype=torch.float32),
            "advantages": torch.as_tensor(adv, dtype=torch.float32),
        }

    def update(self, transitions: List[HighOptionTransition]) -> Dict[str, float]:
        if len(transitions) == 0:
            return {
                "n_samples": 0.0,
                "loss_total": 0.0,
                "loss_actor": 0.0,
                "loss_value": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
            }

        batch = self._build_batch(transitions)
        obs = batch["obs"]
        z = batch["z"]
        old_logp = batch["old_logp"]
        old_value = batch["old_value"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        n = obs.shape[0]

        total_loss_acc = 0.0
        actor_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_acc = 0.0
        approx_kl_acc = 0.0
        clip_frac_acc = 0.0
        n_updates = 0

        for _ in range(self.config.ppo_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.config.minibatch_size):
                idx = perm[start : start + self.config.minibatch_size]
                obs_mb = obs[idx]
                z_mb = z[idx]
                old_logp_mb = old_logp[idx]
                old_value_mb = old_value[idx]
                returns_mb = returns[idx]
                adv_mb = advantages[idx]

                eval_out = self.policy.evaluate_actions(obs_mb, z_mb)
                logp = eval_out["logp"]
                entropy = eval_out["entropy"]
                value = eval_out["value"]

                ratio = torch.exp(logp - old_logp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = old_value_mb + torch.clamp(
                    value - old_value_mb,
                    -self.config.clip_ratio,
                    self.config.clip_ratio,
                )
                value_loss_unclipped = (value - returns_mb) ** 2
                value_loss_clipped = (value_pred_clipped - returns_mb) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_bonus = entropy.mean()
                total_loss = actor_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_bonus

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_logp_mb - logp).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()

                total_loss_acc += float(total_loss.detach().cpu().item())
                actor_loss_acc += float(actor_loss.detach().cpu().item())
                value_loss_acc += float(value_loss.detach().cpu().item())
                entropy_acc += float(entropy_bonus.detach().cpu().item())
                approx_kl_acc += float(approx_kl.detach().cpu().item())
                clip_frac_acc += float(clip_frac.detach().cpu().item())
                n_updates += 1

        scale = 1.0 / max(1, n_updates)
        return {
            "n_samples": float(n),
            "loss_total": total_loss_acc * scale,
            "loss_actor": actor_loss_acc * scale,
            "loss_value": value_loss_acc * scale,
            "entropy": entropy_acc * scale,
            "approx_kl": approx_kl_acc * scale,
            "clip_frac": clip_frac_acc * scale,
        }
