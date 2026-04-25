from __future__ import annotations

import torch
import torch.nn as nn

from openrl.buffers.utils.util import get_critic_obs_space
from openrl.modules.networks.base_value_network import BaseValueNetwork
from openrl.modules.networks.utils.rnn import RNNLayer
from openrl.modules.networks.utils.util import init
from openrl.utils.util import check_v2 as check


class HighLevelCentralizedCritic(BaseValueNetwork):
    """Centralized critic over global high-level state."""

    def __init__(
        self,
        cfg,
        input_space,
        action_space=None,
        use_half: bool = False,
        device=torch.device("cpu"),
        extra_args=None,
    ) -> None:
        super().__init__(cfg, device)
        self.device = device
        self.use_half = use_half
        self.hidden_size = int(getattr(cfg, "high_critic_hidden_size", getattr(cfg, "hidden_size", 128)))
        self._use_orthogonal = bool(getattr(cfg, "use_orthogonal", True))
        self._use_naive_recurrent_policy = bool(getattr(cfg, "use_naive_recurrent_policy", False))
        self._use_recurrent_policy = bool(getattr(cfg, "use_recurrent_policy", False))
        self._recurrent_N = int(getattr(cfg, "recurrent_N", 1))
        self._use_fp16 = bool(getattr(cfg, "use_fp16", False) and getattr(cfg, "use_deepspeed", False))
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_critic_obs_space(input_space)
        if len(obs_shape) != 1:
            raise ValueError(f"HighLevelCentralizedCritic expects 1D Box observations, got shape={obs_shape}")
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

        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        if use_half:
            self.half()
        self.to(device)

    def forward(self, critic_obs, rnn_states, masks):
        critic_obs = check(critic_obs, self.use_half, self.tpdv)
        rnn_states = check(rnn_states, self.use_half, self.tpdv)
        masks = check(masks, self.use_half, self.tpdv)
        if self._use_fp16:
            critic_obs = critic_obs.half()

        features = self.activation(self.fc1(critic_obs))
        features = self.activation(self.fc2(features))
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            features, rnn_states = self.rnn(features, rnn_states, masks)

        values = self.v_out(features)
        return values, rnn_states
