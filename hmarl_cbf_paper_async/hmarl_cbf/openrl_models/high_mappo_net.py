from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

from openrl.modules.base_module import BaseModule
from openrl.modules.common.base_net import BaseNet
from openrl.modules.common.ppo_net import reset_rnn_states
from openrl.modules.ppo_module import PPOModule
from openrl.utils.util import set_seed

from hmarl_cbf.openrl_compat import parse_openrl_default_config
from hmarl_cbf.openrl_models.high_actor import HighLevelActorNetwork
from hmarl_cbf.openrl_models.high_centralized_critic import HighLevelCentralizedCritic


class HighLevelMAPPONet(BaseNet):
    """OpenRL-compatible high-level net with decentralized actor and centralized critic."""

    def __init__(
        self,
        env: Union[gym.Env, str],
        cfg=None,
        device: Union[torch.device, str] = "cpu",
        n_rollout_threads: int = 1,
        model_dict: Optional[Dict[str, Any]] = None,
        module_class: BaseModule = PPOModule,
    ) -> None:
        super().__init__()

        if cfg is None:
            cfg = parse_openrl_default_config()

        set_seed(cfg.seed)
        env.reset(seed=cfg.seed)

        cfg.num_agents = int(env.agent_num)
        cfg.n_rollout_threads = int(n_rollout_threads)
        cfg.learner_n_rollout_threads = int(cfg.n_rollout_threads)
        cfg.use_share_model = False

        if cfg.rnn_type == "gru":
            rnn_hidden_size = cfg.hidden_size
        elif cfg.rnn_type == "lstm":
            rnn_hidden_size = cfg.hidden_size * 2
        else:
            raise NotImplementedError(f"RNN type {cfg.rnn_type} has not been implemented.")
        cfg.rnn_hidden_size = rnn_hidden_size

        if isinstance(device, str):
            device = torch.device(device)

        model_dict = model_dict or {
            "policy": HighLevelActorNetwork,
            "critic": HighLevelCentralizedCritic,
        }

        self.module = module_class(
            cfg=cfg,
            policy_input_space=env.observation_space,
            critic_input_space=env.state_space,
            act_space=env.action_space,
            share_model=False,
            device=device,
            rank=0,
            world_size=1,
            model_dict=model_dict,
        )

        self.cfg = cfg
        self.env = env
        self.device = device
        self.rnn_states_actor = None
        self.masks = None

    def act(
        self,
        observation: np.ndarray | Dict[str, np.ndarray],
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = False,
        episode_starts: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if episode_starts is not None and self.rnn_states_actor is not None:
            self.rnn_states_actor = reset_rnn_states(
                self.rnn_states_actor,
                episode_starts,
                self.env.parallel_env_num,
                self.env.agent_num,
                self.rnn_states_actor.shape[1],
                self.rnn_states_actor.shape[2],
            )

        actions, self.rnn_states_actor = self.module.act(
            obs=observation,
            rnn_states_actor=self.rnn_states_actor,
            masks=self.masks,
            action_masks=action_masks,
            deterministic=deterministic,
        )
        return actions, self.rnn_states_actor

    def reset(self, env: Optional[gym.Env] = None) -> None:
        if env is not None:
            self.env = env
        self.first_reset = False
        self.rnn_states_actor, self.masks = self.module.init_rnn_states(
            rollout_num=self.env.parallel_env_num,
            agent_num=self.env.agent_num,
            rnn_layers=self.cfg.recurrent_N,
            hidden_size=self.cfg.rnn_hidden_size,
        )

    def load_policy(self, path: str) -> None:
        self.module.load_policy(path)
