import pathlib
import typing as T

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from common_utils.tensor_utils import force_pad_batch_size
from data_utils.data_augmentation import move_frame_to_agent_of_idx


def mask_only_target_agents_and_sdc(batch: T.Dict[str, np.ndarray]) -> T.Dict[str, np.ndarray]:
    """Mask the agents that are not the target agents or are not SDCs, it also pads it so that we can concatenate it."""
    target_agents = np.logical_or(batch["tracks_to_predict"], batch["is_sdc"])  # [N_AGENTS,]
    batch["gt_states"] = batch["gt_states"][target_agents]
    batch["gt_states_avails"] = batch["gt_states_avails"][target_agents]
    batch["actor_type"] = batch["actor_type"][target_agents]
    batch["is_sdc"] = batch["is_sdc"][target_agents]
    batch["tracks_to_predict"] = batch["tracks_to_predict"][target_agents]

    MAX_N_AGENTS: T.Final[int] = (
        9  # The maximum number of agents in tracks to predict plus the ego agent
    )
    batch["gt_states"] = force_pad_batch_size(batch["gt_states"], MAX_N_AGENTS)
    batch["gt_states_avails"] = force_pad_batch_size(batch["gt_states_avails"], MAX_N_AGENTS)
    batch["actor_type"] = force_pad_batch_size(batch["actor_type"], MAX_N_AGENTS)
    batch["is_sdc"] = force_pad_batch_size(batch["is_sdc"], MAX_N_AGENTS)
    batch["tracks_to_predict"] = force_pad_batch_size(batch["tracks_to_predict"], MAX_N_AGENTS)
    return batch


def _generate_agent_features(
    gt_states: np.ndarray, gt_states_avails: np.ndarray, delta_t: float = 0.2
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Concatenate history and future data from the batch and compute a state of the form
             {x, y, cos, sin, signed_speed}
    Returns:
     (features, availabilities) [Tensor, Tensor]
    """

    def compute_speed_from_position_diffs(velocities: np.ndarray, yaws: np.ndarray):
        """To compute velocities do integration with the previous timestep.
        s(t+1) = s(t) + v(t) * dt
        v(t) = (s(t+1) - s(t)) / dt
        """
        speed = np.linalg.norm(velocities, axis=-1, keepdims=True)
        speed_sign = np.sign(
            velocities[..., 0, None] * np.cos(yaws) + velocities[..., 1, None] * np.sin(yaws)
        )
        return speed_sign * speed

    positions = gt_states[..., :2]
    length_width = gt_states[..., 2:4]
    yaws = gt_states[..., 4, None]
    velocities = gt_states[..., 5:7]
    availabilities = gt_states_avails.copy()
    speeds = compute_speed_from_position_diffs(velocities, yaws)

    # Get full state representation
    features = np.concatenate(
        (positions, np.cos(yaws), np.sin(yaws), speeds, length_width), axis=-1
    )

    # Make sure to mask out not available agents
    features = features * availabilities[..., None]
    return features.astype(np.float32), availabilities.astype(np.bool_)


def _generate_agent_centric_samples(sample: T.Dict[str, np.ndarray]) -> T.Dict[str, np.ndarray]:
    """Generate multiple samples for each agent in the scenario.
    Args:
        sample (T.Dict[str, np.ndarray]): contains the description of an scenario, including
        agent, map and tl information.
    Returns:
        T.Dict[str, np.ndarray]: adds an extra agent dimension to each tensor in the sample
        and for each available agent it moves the sample to the agent's frame and adds it
        to the output batch.
    """

    from collections import defaultdict

    agent_batch = defaultdict(list)

    MAX_AGENTS = 8
    for agent_idx in range(MAX_AGENTS):
        tracks_to_predict = sample["tracks_to_predict"]
        if not tracks_to_predict[agent_idx]:
            continue

        agent_batch["agent_to_predict"].append(agent_idx)
        agent_sample = move_frame_to_agent_of_idx(agent_idx, sample)
        for key in agent_sample:
            agent_batch[key].append(agent_sample[key])

    for key in agent_batch:
        agent_batch[key] = np.stack(agent_batch[key], axis=0)

    return agent_batch


def _get_scenario_from_h5_file_using_idx(file: T.Mapping[str, np.ndarray], idx: int) -> T.Dict[str, np.ndarray]:
    return {
        "scenario_id": np.array(file["scenario_id"][idx]).astype(np.int64),
        "gt_states": np.array(file["gt_states"][idx]).astype(np.float32),  # [N_AGENTS, TIME, FEATS]
        "gt_states_avails": np.array(file["gt_states_avails"][idx]).astype(
            np.bool_
        ),  # [N_AGENTS, TIME,]
        "actor_type": np.array(file["actor_type"][idx]).astype(np.int64),  # [N_AGENTS,]
        "is_sdc": np.array(file["is_sdc"][idx]).astype(np.bool_),  # [N_AGENTS,]
        "tracks_to_predict": np.array(file["tracks_to_predict"][idx]).astype(np.bool_),  # [N_AGENTS,]
        "roadgraph_features": np.array(file["roadgraph_features"][idx]).astype(
            np.float32
        ),  # [N_POLYLINE, N_POINTS, FEATS]
        "roadgraph_features_mask": np.array(file["roadgraph_features_mask"][idx]).astype(
            np.bool_
        ),  # [N_POLYLINE, N_POINTS]
        "roadgraph_features_types": np.array(file["roadgraph_features_types"][idx]).astype(
            np.int64
        ),  # [N_POLYLINE,]
        "roadgraph_features_ids": np.array(file["roadgraph_features_ids"][idx]).astype(
            np.int16
        ),  # [N_POLYLINE,]
        "tl_states": np.array(file["tl_states"][idx]).astype(np.float32),  # [N_TRAFFIC_LIGHTS, TIME, 2]
        "tl_states_categorical": np.array(file["tl_states_categorical"][idx]).astype(
            np.int64
        ),  # [N_TRAFFIC_LIGHTS, TIME,]
        "tl_avails": np.array(file["tl_avails"][idx]).astype(np.bool_),  # [N_TRAFFIC_LIGHTS, TIME,]
    }


def _get_scenario_from_h5_file(file: T.Mapping[str, np.ndarray]) -> T.Dict[str, np.ndarray]:
    return {
        "scenario_id": np.array(file["scenario_id"]).astype(np.int64),
        "gt_states": np.array(file["gt_states"]).astype(np.float32),  # [N_AGENTS, TIME, FEATS]
        "gt_states_avails": np.array(file["gt_states_avails"]).astype(
            np.bool_
        ),  # [N_AGENTS, TIME,]
        "actor_type": np.array(file["actor_type"]).astype(np.int64),  # [N_AGENTS,]
        "is_sdc": np.array(file["is_sdc"]).astype(np.bool_),  # [N_AGENTS,]
        "tracks_to_predict": np.array(file["tracks_to_predict"]).astype(np.bool_),  # [N_AGENTS,]
        "roadgraph_features": np.array(file["roadgraph_features"]).astype(
            np.float32
        ),  # [N_POLYLINE, N_POINTS, FEATS]
        "roadgraph_features_mask": np.array(file["roadgraph_features_mask"]).astype(
            np.bool_
        ),  # [N_POLYLINE, N_POINTS]
        "roadgraph_features_types": np.array(file["roadgraph_features_types"]).astype(
            np.int64
        ),  # [N_POLYLINE,]
        "roadgraph_features_ids": np.array(file["roadgraph_features_ids"]).astype(
            np.int16
        ),  # [N_POLYLINE,]
        "tl_states": np.array(file["tl_states"]).astype(np.float32),  # [N_TRAFFIC_LIGHTS, TIME, 2]
        "tl_states_categorical": np.array(file["tl_states_categorical"]).astype(
            np.int64
        ),  # [N_TRAFFIC_LIGHTS, TIME,]
        "tl_avails": np.array(file["tl_avails"]).astype(np.bool_),  # [N_TRAFFIC_LIGHTS, TIME,]
    }


class ProcessedDataset(Dataset):
    def __init__(
        self,
        filepath: str,
    ):
        """Do not open the h5 file here"""
        self.filepath = filepath
        self.file: T.Optional[h5py.File] = None

        # Open and close dataset just to extract the length
        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as file:
            self.dataset_len = len(file["gt_states"])  # Example version of the dataset


    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, scenario_idx: int) -> T.Dict[str, np.ndarray]:
        # NOTE: We open the dataset here so that each worker has its own file handle
        if self.file is None:
            self.file = h5py.File(self.filepath, "r", libver="latest", swmr=True)

        sample = _get_scenario_from_h5_file_using_idx(self.file, scenario_idx)

        # agent_batch = _generate_agent_centric_samples(sample)
        return sample 

    def __del__(self):
        if self.file is not None:
            self.file.close()


class ScenarioDataset(Dataset):
    def __init__(
        self,
        filepath: str,
    ):
        self.datadir = pathlib.Path(filepath)
        self.files = list(self.datadir.glob("scenario*.pt"))
        self.coupled_indices = torch.load(self.datadir / "coupled_indices.pt")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, scenario_idx: int) -> T.Dict[str, np.ndarray]:
        sample = torch.load(self.datadir / f"batch_{scenario_idx}.pt")
        agent_indices = self.coupled_indices[self.coupled_indices[:, 0] == scenario_idx][:, 1]

        sample = _get_scenario_from_h5_file(sample)

        batches = []
        for agent_idx in agent_indices:
            agent_sample = move_frame_to_agent_of_idx(agent_idx, sample)

            # TODO: Move to feature generation
            agent_sample["gt_features"], agent_sample["gt_features_avails"] = (
                _generate_agent_features(
                    agent_sample["gt_states"], agent_sample["gt_states_avails"]
                )
            )

            agent_sample["agent_to_predict"] = np.array(agent_idx).astype(np.int64)
            batches.append(agent_sample)

        return batches


class AgentCentricDataset(Dataset):
    def __init__(
        self,
        filepath: str,
    ):
        # Get all .pt files in the directory
        self.datadir = pathlib.Path(filepath)
        self.coupled_indices = torch.load(self.datadir / "coupled_indices.pt")

    def __len__(self) -> int:
        return len(self.coupled_indices)

    def __getitem__(self, idx: int) -> T.Dict[str, np.ndarray]:
        scenario_idx, agent_idx = self.coupled_indices[idx]

        sample = torch.load(self.datadir / f"batch_{scenario_idx}.pt")
        sample = _get_scenario_from_h5_file(sample)
        sample = move_frame_to_agent_of_idx(agent_idx, sample)

        # TODO: Move to feature generation
        sample["gt_features"], sample["gt_features_avails"] = _generate_agent_features(
            sample["gt_states"], sample["gt_states_avails"]
        )

        sample["agent_to_predict"] = np.array(agent_idx).astype(np.int64)
        return sample
