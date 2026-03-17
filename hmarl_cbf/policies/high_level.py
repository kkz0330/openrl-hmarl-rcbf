from __future__ import annotations

from typing import Dict, Tuple

try:
    import torch
    from torch import Tensor
    from torch import nn
    from torch.distributions import Categorical
except ImportError:  # pragma: no cover - import-safe fallback
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc, assignment]
    nn = object  # type: ignore[assignment]
    Categorical = None  # type: ignore[assignment]


class HighLevelPolicy(nn.Module):  # type: ignore[misc]
    """Discrete option policy + value head for synchronous MAPPO."""

    def __init__(self, obs_dim: int, n_skills: int, hidden_dim: int = 128) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to instantiate HighLevelPolicy")
        super().__init__()
        self.obs_dim = obs_dim
        self.n_skills = n_skills
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, n_skills)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_high: Tensor) -> Tuple[Tensor, Tensor]:
        feat = self.backbone(obs_high)
        logits = self.actor_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value

    def act(self, obs_high: Tensor, deterministic: bool = False) -> Dict[str, Tensor]:
        logits, value = self.forward(obs_high)
        dist = Categorical(logits=logits)
        if deterministic:
            z = torch.argmax(logits, dim=-1)
        else:
            z = dist.sample()
        logp = dist.log_prob(z)
        return {"z": z, "logp": logp, "value": value}

    def evaluate_actions(self, obs_high: Tensor, z: Tensor) -> Dict[str, Tensor]:
        logits, value = self.forward(obs_high)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(z)
        entropy = dist.entropy()
        return {"logp": logp, "entropy": entropy, "value": value}
