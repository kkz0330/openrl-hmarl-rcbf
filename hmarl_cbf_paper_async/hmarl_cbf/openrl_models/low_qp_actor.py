from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from openrl.buffers.utils.util import get_policy_obs, get_policy_obs_space
from openrl.modules.networks.base_policy_network import BasePolicyNetwork
from openrl.modules.networks.utils.rnn import RNNLayer
from openrl.modules.networks.utils.util import init
from openrl.utils.util import check_v2 as check


class LowQPActorNetwork(BasePolicyNetwork):
    """Low-level actor that outputs a stochastic phi distribution.

    The actor keeps the original algorithm structure:
    - local low-level observation is split into physical obs + skill one-hot
    - skill is embedded rather than treated as raw one-hot only
    - output is state-dependent ``mu`` and ``log_std`` for phi
    """

    def __init__(
        self,
        cfg,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> None:
        if not hasattr(cfg, "use_valuenorm"):
            setattr(cfg, "use_valuenorm", False)
        if not hasattr(cfg, "use_policy_vhead"):
            setattr(cfg, "use_policy_vhead", False)
        super().__init__(cfg, device)
        extra_args = dict(extra_args or {})
        self.device = device
        self.use_half = use_half
        self._use_naive_recurrent_policy = bool(getattr(cfg, "use_naive_recurrent_policy", False))
        self._use_recurrent_policy = bool(getattr(cfg, "use_recurrent_policy", False))
        self._recurrent_N = int(getattr(cfg, "recurrent_N", 1))
        self._use_fp16 = bool(getattr(cfg, "use_fp16", False) and getattr(cfg, "use_deepspeed", False))
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._use_policy_active_masks = False

        self.n_skills = int(extra_args.get("n_skills", getattr(cfg, "n_skills", 0)))
        if self.n_skills <= 0:
            raise ValueError("LowQPActorNetwork requires positive n_skills in extra_args or cfg")
        self.hidden_dim = int(extra_args.get("hidden_dim", getattr(cfg, "low_hidden_size", getattr(cfg, "hidden_size", 128))))
        self.phi_log_std_min = float(extra_args.get("phi_log_std_min", -5.0))
        self.phi_log_std_max = float(extra_args.get("phi_log_std_max", 1.0))

        obs_shape = get_policy_obs_space(input_space)
        if len(obs_shape) != 1:
            raise ValueError(f"LowQPActorNetwork expects 1D Box observations, got shape={obs_shape}")
        total_obs_dim = int(obs_shape[0])
        if total_obs_dim <= self.n_skills:
            raise ValueError("low-level actor observation dimension must exceed n_skills")
        self.obs_dim = total_obs_dim - self.n_skills
        self.phi_dim = int(action_space.shape[0])

        init_method = nn.init.orthogonal_ if bool(getattr(cfg, "use_orthogonal", True)) else nn.init.xavier_uniform_

        def init_(module: nn.Module, gain: float = 1.0) -> nn.Module:
            return init(module, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.skill_embedding = nn.Embedding(self.n_skills, self.hidden_dim)
        self.obs_encoder = nn.Sequential(
            init_(nn.Linear(self.obs_dim, self.hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            init_(nn.Linear(self.hidden_dim * 2, self.hidden_dim)),
            nn.ReLU(),
        )
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_dim,
                self.hidden_dim,
                self._recurrent_N,
                bool(getattr(cfg, "use_orthogonal", True)),
                rnn_type=getattr(cfg, "rnn_type", "gru"),
            )
        self.phi_mu_head = init_(nn.Linear(self.hidden_dim, self.phi_dim), gain=0.01)
        self.phi_log_std_head = init_(nn.Linear(self.hidden_dim, self.phi_dim), gain=0.01)

        if use_half:
            self.half()
        self.to(device)

    def _split_obs(self, raw_obs):
        obs = get_policy_obs(raw_obs)
        obs = check(obs, self.use_half, self.tpdv)
        if self._use_fp16:
            obs = obs.half()
        base_obs = obs[:, : self.obs_dim]
        skill_one_hot = obs[:, self.obs_dim :]
        skill_id = torch.argmax(skill_one_hot, dim=-1).long()
        return base_obs, skill_id

    def _forward_features(self, raw_obs, rnn_states, masks):
        base_obs, skill_id = self._split_obs(raw_obs)
        rnn_states = check(rnn_states, self.use_half, self.tpdv)
        masks = check(masks, self.use_half, self.tpdv)
        obs_feat = self.obs_encoder(base_obs)
        skill_feat = self.skill_embedding(skill_id)
        features = self.fusion(torch.cat([obs_feat, skill_feat], dim=-1))
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            features, rnn_states = self.rnn(features, rnn_states, masks)
        return features, rnn_states

    def phi_distribution(self, raw_obs, rnn_states, masks):
        features, rnn_states = self._forward_features(raw_obs, rnn_states, masks)
        mu = self.phi_mu_head(features)
        log_std = torch.clamp(
            self.phi_log_std_head(features),
            min=self.phi_log_std_min,
            max=self.phi_log_std_max,
        )
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        return dist, mu, log_std, rnn_states

    def sample_phi(self, raw_obs, rnn_states, masks, deterministic: bool = False):
        dist, mu, log_std, rnn_states = self.phi_distribution(raw_obs, rnn_states, masks)
        phi = mu if deterministic else dist.rsample()
        logp = dist.log_prob(phi).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1).mean()
        return {
            "phi": phi,
            "mu": mu,
            "log_std": log_std,
            "logp": logp,
            "entropy": entropy,
            "rnn_states": rnn_states,
        }

    def forward(self, forward_type, *args, **kwargs):
        if forward_type == "original":
            return self.forward_original(*args, **kwargs)
        if forward_type == "eval_actions":
            return self.eval_actions(*args, **kwargs)
        raise NotImplementedError(forward_type)

    def forward_original(self, raw_obs, rnn_states, masks, action_masks=None, deterministic: bool = False):
        del action_masks
        sample = self.sample_phi(raw_obs, rnn_states, masks, deterministic=deterministic)
        return sample["phi"], sample["logp"], sample["rnn_states"]

    def eval_actions(self, obs, rnn_states, action, masks, action_masks=None, active_masks=None):
        del action_masks, active_masks
        action = check(action, self.use_half, self.tpdv)
        dist, _, _, _ = self.phi_distribution(obs, rnn_states, masks)
        action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(dim=-1).mean()
        return action_log_probs, dist_entropy, None
