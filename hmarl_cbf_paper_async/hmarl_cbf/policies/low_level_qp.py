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
    """Maps (obs_low, skill_id) -> stochastic phi, decoded into full-SPD H and F."""

    def __init__(
        self,
        obs_dim: int,
        n_skills: int,
        action_dim: int = 2,
        hidden_dim: int = 128,
        h_diag_min: float = 1e-2,
        h_diag_max: float = 50.0,
        h_offdiag_abs_max: float = 5.0,
        f_abs_max: float = 20.0,
        phi_log_std_min: float = -5.0,
        phi_log_std_max: float = 1.0,
        w_clf: float = 10.0,
        w_cbf: float = 100.0,
        cbf_slack_max: float = 1.0,
        cbf_k0: float = 1.0,
        cbf_k1: float = 1.0,
        clf_k: float = 1.0,
        hocbf_gamma_h: float = 1.0,
        hocbf_gamma_hdot: float = 1.0,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to instantiate LowLevelQPPolicy")
        super().__init__()
        self.obs_dim = obs_dim
        self.n_skills = n_skills
        self.action_dim = action_dim
        self.h_diag_min = float(h_diag_min)
        self.h_diag_max = float(h_diag_max)
        self.h_offdiag_abs_max = float(h_offdiag_abs_max)
        self.f_abs_max = float(f_abs_max)
        self.phi_log_std_min = float(phi_log_std_min)
        self.phi_log_std_max = float(phi_log_std_max)
        if self.h_diag_min <= 0.0:
            raise ValueError("h_diag_min must be > 0 for strict positive definiteness")
        if self.h_diag_max < self.h_diag_min:
            raise ValueError("h_diag_max must be >= h_diag_min")
        if self.h_offdiag_abs_max < 0.0:
            raise ValueError("h_offdiag_abs_max must be >= 0")
        if self.f_abs_max <= 0.0:
            raise ValueError("f_abs_max must be > 0")

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
        self.phi_dim = (action_dim * (action_dim + 1)) // 2 + action_dim
        self.phi_mu_head = nn.Linear(hidden_dim, self.phi_dim)
        self.phi_log_std_head = nn.Linear(hidden_dim, self.phi_dim)
        self.low_value_head = nn.Linear(hidden_dim, 1)

        self.register_buffer("_fixed_w_clf", torch.tensor([float(w_clf)], dtype=torch.float32))
        self.register_buffer("_fixed_w_cbf", torch.tensor([float(w_cbf)], dtype=torch.float32))
        self.register_buffer("_fixed_cbf_slack_max", torch.tensor([float(cbf_slack_max)], dtype=torch.float32))
        self.register_buffer("_fixed_cbf_k0", torch.tensor([float(cbf_k0)], dtype=torch.float32))
        self.register_buffer("_fixed_cbf_k1", torch.tensor([float(cbf_k1)], dtype=torch.float32))
        self.register_buffer("_fixed_clf_k", torch.tensor([float(clf_k)], dtype=torch.float32))
        self.register_buffer("_fixed_hocbf_gamma_h", torch.tensor([float(hocbf_gamma_h)], dtype=torch.float32))
        self.register_buffer("_fixed_hocbf_gamma_hdot", torch.tensor([float(hocbf_gamma_hdot)], dtype=torch.float32))

    def _encode(self, obs_low: Tensor, skill_id: Tensor) -> Tensor:
        skill_feat = self.skill_embedding(skill_id)
        obs_feat = self.obs_encoder(obs_low)
        return self.fusion(torch.cat([obs_feat, skill_feat], dim=-1))

    def _build_spd_h(self, chol_vec: Tensor) -> Tensor:
        batch = chol_vec.shape[0]
        L = torch.zeros((batch, self.action_dim, self.action_dim), dtype=chol_vec.dtype, device=chol_vec.device)
        cursor = 0
        for row in range(self.action_dim):
            for col in range(row + 1):
                raw = chol_vec[:, cursor]
                if row == col:
                    diag = torch.clamp(
                        F.softplus(raw) + self.h_diag_min,
                        min=self.h_diag_min,
                        max=self.h_diag_max,
                    )
                    L[:, row, col] = diag
                else:
                    if self.h_offdiag_abs_max > 0.0:
                        L[:, row, col] = self.h_offdiag_abs_max * torch.tanh(raw)
                    else:
                        L[:, row, col] = torch.zeros_like(raw)
                cursor += 1
        H = torch.matmul(L, L.transpose(-1, -2))
        jitter = self.h_diag_min * torch.eye(self.action_dim, dtype=H.dtype, device=H.device).unsqueeze(0)
        return H + jitter

    def _phi_distribution(self, obs_low: Tensor, skill_id: Tensor) -> tuple[Tensor, Tensor]:
        fused = self._encode(obs_low, skill_id)
        mu = self.phi_mu_head(fused)
        log_std = torch.clamp(
            self.phi_log_std_head(fused),
            min=self.phi_log_std_min,
            max=self.phi_log_std_max,
        )
        return mu, log_std

    def decode_phi(self, phi: Tensor) -> QPParam:
        batch = phi.shape[0]
        chol_dim = (self.action_dim * (self.action_dim + 1)) // 2
        chol_vec = phi[:, :chol_dim]
        f_raw = phi[:, chol_dim:]
        H_mat = self._build_spd_h(chol_vec)
        f_lin = self.f_abs_max * torch.tanh(f_raw)
        return QPParam(
            H_mat=H_mat,
            f_lin=f_lin,
            w_clf=self._fixed_w_clf.expand(batch, -1),
            w_cbf=self._fixed_w_cbf.expand(batch, -1),
            cbf_slack_max=self._fixed_cbf_slack_max.expand(batch, -1),
            cbf_k0=self._fixed_cbf_k0.expand(batch, -1),
            cbf_k1=self._fixed_cbf_k1.expand(batch, -1),
            clf_k=self._fixed_clf_k.expand(batch, -1),
            hocbf_gamma_h=self._fixed_hocbf_gamma_h.expand(batch, -1),
            hocbf_gamma_hdot=self._fixed_hocbf_gamma_hdot.expand(batch, -1),
        )

    def sample_qp_params(self, obs_low: Tensor, skill_id: Tensor, deterministic: bool = False) -> dict[str, Tensor | QPParam]:
        mu, log_std = self._phi_distribution(obs_low, skill_id)
        std = torch.exp(log_std)
        if deterministic:
            phi = mu
            logp = torch.zeros((mu.shape[0],), dtype=mu.dtype, device=mu.device)
            entropy = torch.zeros((mu.shape[0],), dtype=mu.dtype, device=mu.device)
        else:
            dist = torch.distributions.Normal(mu, std)
            phi = dist.rsample()
            logp = torch.sum(dist.log_prob(phi), dim=-1)
            entropy = torch.sum(dist.entropy(), dim=-1)
        qp_param = self.decode_phi(phi)
        return {
            "phi": phi,
            "mu": mu,
            "log_std": log_std,
            "logp": logp,
            "entropy": entropy,
            "qp_param": qp_param,
        }

    def evaluate_phi(self, obs_low: Tensor, skill_id: Tensor, phi: Tensor) -> dict[str, Tensor | QPParam]:
        mu, log_std = self._phi_distribution(obs_low, skill_id)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        logp = torch.sum(dist.log_prob(phi), dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)
        qp_param = self.decode_phi(phi)
        return {
            "mu": mu,
            "log_std": log_std,
            "logp": logp,
            "entropy": entropy,
            "qp_param": qp_param,
        }

    def forward(self, obs_low: Tensor, skill_id: Tensor) -> QPParam:
        mu, _ = self._phi_distribution(obs_low, skill_id)
        return self.decode_phi(mu)

    def low_value(self, obs_low: Tensor, skill_id: Tensor) -> Tensor:
        fused = self._encode(obs_low, skill_id)
        return self.low_value_head(fused).reshape(-1)
