from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def dict_obs_to_stacked(agent_ids: Sequence[int], obs: Mapping[int, np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(obs[int(agent_id)], dtype=np.float32).reshape(-1) for agent_id in agent_ids], axis=0)


def dict_scalar_to_stacked(agent_ids: Sequence[int], values: Mapping[int, float]) -> np.ndarray:
    return np.asarray([float(values.get(int(agent_id), 0.0)) for agent_id in agent_ids], dtype=np.float32)


def dict_mask_to_stacked(agent_ids: Sequence[int], masks: Mapping[int, np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(masks[int(agent_id)], dtype=np.float32).reshape(-1) for agent_id in agent_ids], axis=0)


def dict_bool_to_stacked(agent_ids: Sequence[int], values: Mapping[int, bool]) -> np.ndarray:
    return np.asarray([bool(values.get(int(agent_id), False)) for agent_id in agent_ids], dtype=np.bool_)


def stacked_actions_to_dict(agent_ids: Sequence[int], actions: np.ndarray | Iterable[int]) -> Dict[int, int]:
    arr = np.asarray(list(actions) if not isinstance(actions, np.ndarray) else actions).reshape(-1)
    return {int(agent_id): int(arr[idx]) for idx, agent_id in enumerate(agent_ids)}


def merge_infos_to_openrl_env_info(agent_ids: Sequence[int], infos: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    first_info = infos[int(agent_ids[0])] if len(agent_ids) > 0 else {}
    action_masks = dict_mask_to_stacked(
        agent_ids,
        {int(agent_id): np.asarray(infos[int(agent_id)].get("action_masks", 0.0), dtype=np.float32) for agent_id in agent_ids},
    )
    return {
        "action_masks": action_masks,
        "critic_state": np.asarray(first_info.get("critic_state", np.array([], dtype=np.float32)), dtype=np.float32).copy(),
        "switch_required": dict_bool_to_stacked(
            agent_ids,
            {int(agent_id): bool(infos[int(agent_id)].get("switch_required", infos[int(agent_id)].get("switch_required_next", False))) for agent_id in agent_ids},
        ),
        "raw_infos": {int(agent_id): dict(infos[int(agent_id)]) for agent_id in agent_ids},
    }


def reset_to_openrl_batch(
    agent_ids: Sequence[int],
    obs: Mapping[int, np.ndarray],
    infos: Mapping[int, Mapping[str, Any]],
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    return dict_obs_to_stacked(agent_ids, obs), [merge_infos_to_openrl_env_info(agent_ids, infos)]


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
        [merge_infos_to_openrl_env_info(agent_ids, infos)],
    )
