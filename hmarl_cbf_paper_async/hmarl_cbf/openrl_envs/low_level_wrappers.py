from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from hmarl_cbf.openrl_envs.high_level_wrappers import (
    dict_bool_to_stacked,
    dict_obs_to_stacked,
    dict_scalar_to_stacked,
)


def continuous_stacked_actions_to_dict(agent_ids: Sequence[int], actions: np.ndarray | Iterable[np.ndarray]) -> Dict[int, np.ndarray]:
    arr = np.asarray(list(actions) if not isinstance(actions, np.ndarray) else actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(len(agent_ids), -1)
    return {int(agent_id): np.asarray(arr[idx], dtype=np.float32).reshape(-1) for idx, agent_id in enumerate(agent_ids)}


def merge_low_infos_to_openrl_env_info(agent_ids: Sequence[int], infos: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    first_info = infos[int(agent_ids[0])] if len(agent_ids) > 0 else {}
    return {
        "critic_state": np.asarray(first_info.get("critic_state", np.array([], dtype=np.float32)), dtype=np.float32).copy(),
        "skill_ids": np.asarray([int(infos[int(agent_id)].get("skill_id", 0)) for agent_id in agent_ids], dtype=np.int64),
        "pending_switch_agents": np.asarray(first_info.get("pending_switch_agents", np.array([], dtype=np.int64)), dtype=np.int64).copy(),
        "switch_required": dict_bool_to_stacked(
            agent_ids,
            {int(agent_id): bool(infos[int(agent_id)].get("switch_required", False)) for agent_id in agent_ids},
        ),
        "terminated_by_skill": dict_bool_to_stacked(
            agent_ids,
            {int(agent_id): bool(infos[int(agent_id)].get("terminated_by_skill", False)) for agent_id in agent_ids},
        ),
        "intrinsic_reward": dict_scalar_to_stacked(
            agent_ids,
            {int(agent_id): float(infos[int(agent_id)].get("intrinsic_reward", 0.0)) for agent_id in agent_ids},
        ),
        "reward_ext": dict_scalar_to_stacked(
            agent_ids,
            {int(agent_id): float(infos[int(agent_id)].get("reward_ext", 0.0)) for agent_id in agent_ids},
        ),
        "raw_infos": {int(agent_id): dict(infos[int(agent_id)]) for agent_id in agent_ids},
    }


def reset_to_openrl_batch(
    agent_ids: Sequence[int],
    obs: Mapping[int, np.ndarray],
    infos: Mapping[int, Mapping[str, Any]],
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    return dict_obs_to_stacked(agent_ids, obs), [merge_low_infos_to_openrl_env_info(agent_ids, infos)]


def step_to_openrl_batch(
    agent_ids: Sequence[int],
    obs: Mapping[int, np.ndarray],
    rewards: Mapping[int, float],
    terminations: Mapping[int, bool],
    truncations: Mapping[int, bool],
    infos: Mapping[int, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    return (
        dict_obs_to_stacked(agent_ids, obs),
        dict_scalar_to_stacked(agent_ids, rewards),
        dict_bool_to_stacked(agent_ids, terminations),
        dict_bool_to_stacked(agent_ids, truncations),
        [merge_low_infos_to_openrl_env_info(agent_ids, infos)],
    )
