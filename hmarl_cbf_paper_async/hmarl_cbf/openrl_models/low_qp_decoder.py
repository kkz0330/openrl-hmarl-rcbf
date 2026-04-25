from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hmarl_cbf.types import QPParam


class LowQPDecoder(nn.Module):
    """Decode continuous phi into structured QP parameters.

    This module preserves the low-level algorithm logic:
    - phi parameterizes a full SPD H and linear term f
    - f can optionally be residual-on-reference
    - CBF/CLF scalars and safety radii can optionally be parameterized by phi
    """

    _CBF_PARAM_DIM = 7

    @classmethod
    def compute_phi_dim(cls, action_dim: int, *, parameterize_cbf_constraints: bool = False) -> int:
        base_dim = (int(action_dim) * (int(action_dim) + 1)) // 2 + int(action_dim)
        return base_dim + (cls._CBF_PARAM_DIM if parameterize_cbf_constraints else 0)

    def __init__(
        self,
        *,
        action_dim: int = 2,
        h_diag_min: float = 1e-2,
        h_diag_max: float = 50.0,
        h_offdiag_abs_max: float = 5.0,
        f_abs_max: float = 20.0,
        w_clf: float = 10.0,
        w_cbf: float = 100.0,
        cbf_slack_max: float = 1.0,
        cbf_k0: float = 1.0,
        cbf_k1: float = 1.0,
        clf_k: float = 1.0,
        hocbf_gamma_h: float = 1.0,
        hocbf_gamma_hdot: float = 1.0,
        d_min_agent: float = 0.6,
        d_safe_obs: float = 0.6,
        parameterize_cbf_constraints: bool = False,
        cbf_k0_min: float = 0.0,
        cbf_k0_max: float = 2.0,
        cbf_k1_min: float = 0.0,
        cbf_k1_max: float = 2.0,
        clf_k_min: float = 0.0,
        clf_k_max: float = 2.0,
        hocbf_gamma_h_min: float = 0.0,
        hocbf_gamma_h_max: float = 2.0,
        hocbf_gamma_hdot_min: float = 0.0,
        hocbf_gamma_hdot_max: float = 2.0,
        d_min_agent_min: float = 0.05,
        d_min_agent_max: float = 0.6,
        d_safe_obs_min: float = 0.05,
        d_safe_obs_max: float = 0.6,
        f_residual_reference_enabled: bool = True,
        f_ref_speed: float = 1.2,
        f_ref_kp: float = 1.2,
        f_ref_slow_radius: float = 1.5,
        f_ref_goal_stop_min_speed: float = 0.0,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if h_diag_min <= 0.0:
            raise ValueError("h_diag_min must be > 0")
        if h_diag_max < h_diag_min:
            raise ValueError("h_diag_max must be >= h_diag_min")
        if h_offdiag_abs_max < 0.0:
            raise ValueError("h_offdiag_abs_max must be >= 0")
        if f_abs_max <= 0.0:
            raise ValueError("f_abs_max must be > 0")
        if d_min_agent <= 0.0:
            raise ValueError("d_min_agent must be > 0")
        if d_safe_obs <= 0.0:
            raise ValueError("d_safe_obs must be > 0")

        self.action_dim = int(action_dim)
        self.parameterize_cbf_constraints = bool(parameterize_cbf_constraints)
        self.phi_dim = self.compute_phi_dim(
            self.action_dim,
            parameterize_cbf_constraints=self.parameterize_cbf_constraints,
        )
        self.h_diag_min = float(h_diag_min)
        self.h_diag_max = float(h_diag_max)
        self.h_offdiag_abs_max = float(h_offdiag_abs_max)
        self.f_abs_max = float(f_abs_max)
        self.f_residual_reference_enabled = bool(f_residual_reference_enabled)
        self.f_ref_speed = float(f_ref_speed)
        self.f_ref_kp = float(f_ref_kp)
        self.f_ref_slow_radius = float(f_ref_slow_radius)
        self.f_ref_goal_stop_min_speed = float(f_ref_goal_stop_min_speed)

        def _validate_range(name: str, lo: float, hi: float) -> tuple[float, float]:
            lo_f = float(lo)
            hi_f = float(hi)
            if lo_f < 0.0:
                raise ValueError(f"{name}_min must be >= 0")
            if hi_f < lo_f:
                raise ValueError(f"{name}_max must be >= {name}_min")
            return lo_f, hi_f

        cbf_k0_min, cbf_k0_max = _validate_range("cbf_k0", cbf_k0_min, cbf_k0_max)
        cbf_k1_min, cbf_k1_max = _validate_range("cbf_k1", cbf_k1_min, cbf_k1_max)
        clf_k_min, clf_k_max = _validate_range("clf_k", clf_k_min, clf_k_max)
        hocbf_gamma_h_min, hocbf_gamma_h_max = _validate_range("hocbf_gamma_h", hocbf_gamma_h_min, hocbf_gamma_h_max)
        hocbf_gamma_hdot_min, hocbf_gamma_hdot_max = _validate_range(
            "hocbf_gamma_hdot",
            hocbf_gamma_hdot_min,
            hocbf_gamma_hdot_max,
        )
        d_min_agent_min, d_min_agent_max = _validate_range("d_min_agent", d_min_agent_min, d_min_agent_max)
        d_safe_obs_min, d_safe_obs_max = _validate_range("d_safe_obs", d_safe_obs_min, d_safe_obs_max)

        self.register_buffer("_fixed_w_clf", torch.tensor([float(w_clf)], dtype=torch.float32))
        self.register_buffer("_fixed_w_cbf", torch.tensor([float(w_cbf)], dtype=torch.float32))
        self.register_buffer("_fixed_cbf_slack_max", torch.tensor([float(cbf_slack_max)], dtype=torch.float32))
        self.register_buffer("_fixed_cbf_k0", torch.tensor([float(cbf_k0)], dtype=torch.float32))
        self.register_buffer("_fixed_cbf_k1", torch.tensor([float(cbf_k1)], dtype=torch.float32))
        self.register_buffer("_fixed_clf_k", torch.tensor([float(clf_k)], dtype=torch.float32))
        self.register_buffer("_fixed_hocbf_gamma_h", torch.tensor([float(hocbf_gamma_h)], dtype=torch.float32))
        self.register_buffer("_fixed_hocbf_gamma_hdot", torch.tensor([float(hocbf_gamma_hdot)], dtype=torch.float32))
        self.register_buffer("_fixed_d_min_agent", torch.tensor([float(d_min_agent)], dtype=torch.float32))
        self.register_buffer("_fixed_d_safe_obs", torch.tensor([float(d_safe_obs)], dtype=torch.float32))
        self.register_buffer("_cbf_k0_min", torch.tensor([cbf_k0_min], dtype=torch.float32))
        self.register_buffer("_cbf_k0_max", torch.tensor([cbf_k0_max], dtype=torch.float32))
        self.register_buffer("_cbf_k1_min", torch.tensor([cbf_k1_min], dtype=torch.float32))
        self.register_buffer("_cbf_k1_max", torch.tensor([cbf_k1_max], dtype=torch.float32))
        self.register_buffer("_clf_k_min", torch.tensor([clf_k_min], dtype=torch.float32))
        self.register_buffer("_clf_k_max", torch.tensor([clf_k_max], dtype=torch.float32))
        self.register_buffer("_hocbf_gamma_h_min", torch.tensor([hocbf_gamma_h_min], dtype=torch.float32))
        self.register_buffer("_hocbf_gamma_h_max", torch.tensor([hocbf_gamma_h_max], dtype=torch.float32))
        self.register_buffer("_hocbf_gamma_hdot_min", torch.tensor([hocbf_gamma_hdot_min], dtype=torch.float32))
        self.register_buffer("_hocbf_gamma_hdot_max", torch.tensor([hocbf_gamma_hdot_max], dtype=torch.float32))
        self.register_buffer("_d_min_agent_min", torch.tensor([d_min_agent_min], dtype=torch.float32))
        self.register_buffer("_d_min_agent_max", torch.tensor([d_min_agent_max], dtype=torch.float32))
        self.register_buffer("_d_safe_obs_min", torch.tensor([d_safe_obs_min], dtype=torch.float32))
        self.register_buffer("_d_safe_obs_max", torch.tensor([d_safe_obs_max], dtype=torch.float32))
        self.register_buffer(
            "_cbf_k0_init_raw",
            torch.tensor([self._logit_clamped(cbf_k0, cbf_k0_min, cbf_k0_max)], dtype=torch.float32),
        )
        self.register_buffer(
            "_cbf_k1_init_raw",
            torch.tensor([self._logit_clamped(cbf_k1, cbf_k1_min, cbf_k1_max)], dtype=torch.float32),
        )
        self.register_buffer(
            "_clf_k_init_raw",
            torch.tensor([self._logit_clamped(clf_k, clf_k_min, clf_k_max)], dtype=torch.float32),
        )
        self.register_buffer(
            "_hocbf_gamma_h_init_raw",
            torch.tensor([self._logit_clamped(hocbf_gamma_h, hocbf_gamma_h_min, hocbf_gamma_h_max)], dtype=torch.float32),
        )
        self.register_buffer(
            "_hocbf_gamma_hdot_init_raw",
            torch.tensor(
                [self._logit_clamped(hocbf_gamma_hdot, hocbf_gamma_hdot_min, hocbf_gamma_hdot_max)],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "_d_min_agent_init_raw",
            torch.tensor([self._logit_clamped(d_min_agent, d_min_agent_min, d_min_agent_max)], dtype=torch.float32),
        )
        self.register_buffer(
            "_d_safe_obs_init_raw",
            torch.tensor([self._logit_clamped(d_safe_obs, d_safe_obs_min, d_safe_obs_max)], dtype=torch.float32),
        )

    @property
    def cbf_slack_max_value(self) -> float:
        return float(self._fixed_cbf_slack_max.item())

    def set_cbf_slack_max(self, value: float) -> None:
        self._fixed_cbf_slack_max.fill_(max(0.0, float(value)))

    @staticmethod
    def _box_param(raw: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
        return lo + (hi - lo) * torch.sigmoid(raw)

    @staticmethod
    def _logit_clamped(value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        eps = 1e-4
        scaled = (float(value) - float(lo)) / max(float(hi) - float(lo), 1e-8)
        scaled = min(max(scaled, eps), 1.0 - eps)
        return float(torch.logit(torch.tensor(scaled, dtype=torch.float32)).item())

    @staticmethod
    def _unit(vec: Tensor) -> Tensor:
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
        fallback = torch.zeros_like(vec)
        fallback[..., 0] = 1.0
        return torch.where(norm > 1e-8, vec / torch.clamp(norm, min=1e-8), fallback)

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

    def _f_reference(self, obs_low: Tensor, H_mat: Tensor) -> Tensor:
        vel = obs_low[:, 2:4]
        goal_rel = obs_low[:, 4:6]
        goal_dist = torch.linalg.norm(goal_rel, dim=-1, keepdim=True)
        goal_dir = self._unit(goal_rel)

        speed_des = torch.full_like(goal_dist, self.f_ref_speed)
        if self.f_ref_slow_radius > 0.0:
            speed_scale = torch.clamp(goal_dist / max(self.f_ref_slow_radius, 1e-8), min=0.0, max=1.0)
            speed_des = torch.maximum(
                torch.full_like(speed_des, self.f_ref_goal_stop_min_speed),
                speed_des * speed_scale,
            )
        v_des = speed_des * goal_dir
        a_des = self.f_ref_kp * (v_des - vel)
        return -torch.matmul(H_mat, a_des.unsqueeze(-1)).squeeze(-1)

    def forward(self, obs_low: Tensor, phi: Tensor) -> QPParam:
        if obs_low.dim() != 2:
            raise ValueError(f"obs_low must be rank-2, got shape={tuple(obs_low.shape)}")
        if phi.dim() != 2:
            raise ValueError(f"phi must be rank-2, got shape={tuple(phi.shape)}")
        if phi.shape[-1] != self.phi_dim:
            raise ValueError(f"phi last dim must be {self.phi_dim}, got {phi.shape[-1]}")
        if obs_low.shape[0] != phi.shape[0]:
            raise ValueError("obs_low and phi batch sizes must match")

        batch = phi.shape[0]
        chol_dim = (self.action_dim * (self.action_dim + 1)) // 2
        chol_vec = phi[:, :chol_dim]
        f_raw = phi[:, chol_dim:]
        cbf_param_raw = None
        if self.parameterize_cbf_constraints:
            f_raw = phi[:, chol_dim : chol_dim + self.action_dim]
            cbf_param_raw = phi[:, chol_dim + self.action_dim :]
        H_mat = self._build_spd_h(chol_vec)
        f_residual = self.f_abs_max * torch.tanh(f_raw)
        if self.f_residual_reference_enabled:
            f_ref = self._f_reference(obs_low, H_mat)
            f_lin = f_ref + f_residual
        else:
            f_lin = f_residual

        if self.parameterize_cbf_constraints:
            cbf_k0 = self._box_param(cbf_param_raw[:, 0:1] + self._cbf_k0_init_raw, self._cbf_k0_min, self._cbf_k0_max)
            cbf_k1 = self._box_param(cbf_param_raw[:, 1:2] + self._cbf_k1_init_raw, self._cbf_k1_min, self._cbf_k1_max)
            clf_k = self._box_param(cbf_param_raw[:, 2:3] + self._clf_k_init_raw, self._clf_k_min, self._clf_k_max)
            hocbf_gamma_h = self._box_param(
                cbf_param_raw[:, 3:4] + self._hocbf_gamma_h_init_raw,
                self._hocbf_gamma_h_min,
                self._hocbf_gamma_h_max,
            )
            hocbf_gamma_hdot = self._box_param(
                cbf_param_raw[:, 4:5] + self._hocbf_gamma_hdot_init_raw,
                self._hocbf_gamma_hdot_min,
                self._hocbf_gamma_hdot_max,
            )
            d_min_agent = self._box_param(
                cbf_param_raw[:, 5:6] + self._d_min_agent_init_raw,
                self._d_min_agent_min,
                self._d_min_agent_max,
            )
            d_safe_obs = self._box_param(
                cbf_param_raw[:, 6:7] + self._d_safe_obs_init_raw,
                self._d_safe_obs_min,
                self._d_safe_obs_max,
            )
        else:
            cbf_k0 = self._fixed_cbf_k0.expand(batch, -1)
            cbf_k1 = self._fixed_cbf_k1.expand(batch, -1)
            clf_k = self._fixed_clf_k.expand(batch, -1)
            hocbf_gamma_h = self._fixed_hocbf_gamma_h.expand(batch, -1)
            hocbf_gamma_hdot = self._fixed_hocbf_gamma_hdot.expand(batch, -1)
            d_min_agent = self._fixed_d_min_agent.expand(batch, -1)
            d_safe_obs = self._fixed_d_safe_obs.expand(batch, -1)

        return QPParam(
            H_mat=H_mat,
            f_lin=f_lin,
            w_clf=self._fixed_w_clf.expand(batch, -1),
            w_cbf=self._fixed_w_cbf.expand(batch, -1),
            cbf_slack_max=self._fixed_cbf_slack_max.expand(batch, -1),
            cbf_k0=cbf_k0,
            cbf_k1=cbf_k1,
            clf_k=clf_k,
            hocbf_gamma_h=hocbf_gamma_h,
            hocbf_gamma_hdot=hocbf_gamma_hdot,
            d_min_agent=d_min_agent,
            d_safe_obs=d_safe_obs,
        )
