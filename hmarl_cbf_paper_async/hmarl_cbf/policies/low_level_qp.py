from __future__ import annotations

try:
    import torch
    from torch import Tensor
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - import-safe fallback
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc, assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from hmarl_cbf.types import QPParam


class LowLevelQPPolicy(nn.Module):  # type: ignore[misc]
    """Maps (obs_low, skill_id) -> parameterized QP coefficients."""

    def __init__(
        self,
        obs_dim: int,
        n_skills: int,
        action_dim: int = 2,
        hidden_dim: int = 128,
        r_diag_min: float = 1e-2,
        r_diag_max: float = 50.0,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to instantiate LowLevelQPPolicy")
        super().__init__()
        self.obs_dim = obs_dim
        self.n_skills = n_skills
        self.action_dim = action_dim
        self.r_diag_min = float(r_diag_min)
        self.r_diag_max = float(r_diag_max)
        if self.r_diag_min <= 0.0:
            raise ValueError("r_diag_min must be > 0 for strict positive definiteness")
        if self.r_diag_max < self.r_diag_min:
            raise ValueError("r_diag_max must be >= r_diag_min")

        self.skill_embedding = nn.Embedding(n_skills, hidden_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.u_ref_head = nn.Linear(hidden_dim, action_dim)
        self.r_diag_head = nn.Linear(hidden_dim, action_dim)
        self.w_clf_head = nn.Linear(hidden_dim, 1)
        self.cbf_k0_head = nn.Linear(hidden_dim, 1)
        self.cbf_k1_head = nn.Linear(hidden_dim, 1)
        self.clf_k_head = nn.Linear(hidden_dim, 1)
        self.low_value_head = nn.Linear(hidden_dim, 1)

    def _encode(self, obs_low: Tensor, skill_id: Tensor) -> Tensor:
        skill_feat = self.skill_embedding(skill_id)
        obs_feat = self.obs_encoder(obs_low)
        return self.fusion(torch.cat([obs_feat, skill_feat], dim=-1))

    def forward(self, obs_low: Tensor, skill_id: Tensor) -> QPParam:
        fused = self._encode(obs_low, skill_id)

        u_ref = self.u_ref_head(fused)
        r_diag = torch.clamp(
            F.softplus(self.r_diag_head(fused)) + self.r_diag_min,
            min=self.r_diag_min,
            max=self.r_diag_max,
        )
        w_clf = F.softplus(self.w_clf_head(fused)) + 1e-4
        cbf_k0 = F.softplus(self.cbf_k0_head(fused)) + 1e-4
        cbf_k1 = F.softplus(self.cbf_k1_head(fused)) + 1e-4
        clf_k = F.softplus(self.clf_k_head(fused)) + 1e-4
        return QPParam(
            u_ref=u_ref,
            r_diag=r_diag,
            w_clf=w_clf,
            cbf_k0=cbf_k0,
            cbf_k1=cbf_k1,
            clf_k=clf_k,
            f_lin=None,
            hocbf_gamma_h=None,
            hocbf_gamma_hdot=None,
        )

    def low_value(self, obs_low: Tensor, skill_id: Tensor) -> Tensor:
        fused = self._encode(obs_low, skill_id)
        return self.low_value_head(fused).reshape(-1)
