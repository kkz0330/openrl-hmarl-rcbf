from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from openrl.buffers.utils.util import get_policy_obs, get_policy_obs_space
from openrl.modules.networks.base_policy_network import BasePolicyNetwork
from openrl.modules.networks.utils.rnn import RNNLayer
from openrl.modules.networks.utils.util import init
from openrl.utils.util import check_v2 as check


class HighLevelActorNetwork(BasePolicyNetwork):
    """High-level decentralized actor for discrete skill selection."""

    def __init__(
        self,
        cfg,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half: bool = False,
        extra_args=None,
    ) -> None:
        super().__init__(cfg, device)
        self.device = device
        self.use_half = use_half
        self.hidden_size = int(getattr(cfg, "high_actor_hidden_size", getattr(cfg, "hidden_size", 128)))
        self._gain = float(getattr(cfg, "gain", 0.01))
        self._use_orthogonal = bool(getattr(cfg, "use_orthogonal", True))
        self._use_policy_active_masks = bool(getattr(cfg, "use_policy_active_masks", True))
        self._use_naive_recurrent_policy = bool(getattr(cfg, "use_naive_recurrent_policy", False))
        self._use_recurrent_policy = bool(getattr(cfg, "use_recurrent_policy", False))
        self._recurrent_N = int(getattr(cfg, "recurrent_N", 1))
        self._use_fp16 = bool(getattr(cfg, "use_fp16", False) and getattr(cfg, "use_deepspeed", False))
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_policy_obs_space(input_space)
        if len(obs_shape) != 1:
            raise ValueError(f"HighLevelActorNetwork expects 1D Box observations, got shape={obs_shape}")
        obs_dim = int(obs_shape[0])

        init_method = nn.init.orthogonal_ if self._use_orthogonal else nn.init.xavier_uniform_

        def init_(module: nn.Module, gain: float = 1.0) -> nn.Module:
            return init(module, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = init_(nn.Linear(obs_dim, self.hidden_size))
        self.fc2 = init_(nn.Linear(self.hidden_size, self.hidden_size))
        self.activation = nn.Tanh()

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
                rnn_type=getattr(cfg, "rnn_type", "gru"),
            )

        if not hasattr(action_space, "n"):
            raise ValueError("HighLevelActorNetwork requires a discrete action space")
        self.action_dim = int(action_space.n)
        self.logits_head = init_(nn.Linear(self.hidden_size, self.action_dim), gain=self._gain)

        if use_half:
            self.half()
        self.to(device)

    def _forward_features(self, raw_obs, rnn_states, masks):
        obs = get_policy_obs(raw_obs)
        obs = check(obs, self.use_half, self.tpdv)
        if self._use_fp16:
            obs = obs.half()
        rnn_states = check(rnn_states, self.use_half, self.tpdv)
        masks = check(masks, self.use_half, self.tpdv)

        features = self.activation(self.fc1(obs))
        features = self.activation(self.fc2(features))
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            features, rnn_states = self.rnn(features, rnn_states, masks)
        return features, rnn_states

    def forward(self, forward_type, *args, **kwargs):
        if forward_type == "original":
            return self.forward_original(*args, **kwargs)
        if forward_type == "eval_actions":
            return self.eval_actions(*args, **kwargs)
        raise NotImplementedError(forward_type)

    def _masked_logits(self, actor_features: torch.Tensor, action_masks: torch.Tensor | None) -> torch.Tensor:
        logits = self.logits_head(actor_features)
        if action_masks is None:
            return logits
        legal = action_masks > 0.0
        if torch.any(torch.sum(legal.to(dtype=torch.int64), dim=1) <= 0):
            bad_rows = torch.nonzero(torch.sum(legal.to(dtype=torch.int64), dim=1) <= 0, as_tuple=False).reshape(-1)
            raise RuntimeError(f"high-level actor received empty legal action sets for rows {bad_rows.tolist()}")
        mask_fill = torch.finfo(logits.dtype).min
        return torch.where(legal, logits, torch.full_like(logits, mask_fill))

    def action_mask_from_valid_ids(
        self,
        valid_action_ids: list[list[int]],
    ) -> torch.Tensor:
        mask = torch.zeros((len(valid_action_ids), self.action_dim), dtype=torch.float32, device=self.device)
        for row, valid_ids in enumerate(valid_action_ids):
            if not valid_ids:
                raise RuntimeError(f"high-level actor received empty valid_action_ids for row {row}")
            for action_id in valid_ids:
                if 0 <= int(action_id) < self.action_dim:
                    mask[row, int(action_id)] = 1.0
        return mask

    def forward_original(
        self,
        raw_obs,
        rnn_states,
        masks,
        action_masks=None,
        deterministic: bool = False,
    ):
        if action_masks is not None:
            action_masks = check(action_masks, self.use_half, self.tpdv)
        actor_features, rnn_states = self._forward_features(raw_obs, rnn_states, masks)
        logits = self._masked_logits(actor_features, action_masks)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        action_log_probs = dist.log_prob(actions).unsqueeze(-1)
        return actions.unsqueeze(-1), action_log_probs, rnn_states

    def forward_valid_actions(
        self,
        raw_obs,
        rnn_states,
        masks,
        valid_action_ids: list[list[int]],
        deterministic: bool = False,
    ):
        action_masks = self.action_mask_from_valid_ids(valid_action_ids)
        return self.forward_original(
            raw_obs,
            rnn_states,
            masks,
            action_masks=action_masks,
            deterministic=deterministic,
        )

    def eval_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        action_masks=None,
        active_masks=None,
    ):
        action = check(action, self.use_half, self.tpdv)
        if action_masks is not None:
            action_masks = check(action_masks, self.use_half, self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks, self.use_half, self.tpdv)
        actor_features, _ = self._forward_features(obs, rnn_states, masks)
        logits = self._masked_logits(actor_features, action_masks)
        dist = Categorical(logits=logits)
        action_flat = action.long().reshape(-1)
        action_log_probs = dist.log_prob(action_flat).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        if self._use_policy_active_masks and active_masks is not None:
            weights = active_masks.reshape(-1, 1)
            dist_entropy = torch.sum(entropy * weights) / torch.clamp(torch.sum(weights), min=1e-6)
        else:
            dist_entropy = entropy.mean()
        return action_log_probs, dist_entropy, None
