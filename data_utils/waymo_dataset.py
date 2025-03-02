import copy
import pathlib
import pickle
import typing as T

import numpy as np
from torch.utils.data import Dataset

from data_utils.feature_description import NUM_HISTORY_FRAMES
from utils.geometry import (
    get_so2_from_se2,
    get_transformation_matrix,
    get_yaw_from_se2,
    transform_points,
)


def _generate_agent_features(
    gt_states: np.ndarray, gt_states_avails: np.ndarray, delta_t: float = 0.2
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Concatenate history and future data from the batch and compute a state of the form
             {x, y, cos, sin, signed_speed}
    Returns:
     (features, availabilities) [Tensor, Tensor]
    """

    def compute_speed_from_position_diffs(
        velocities: np.ndarray, yaws: np.ndarray
    ) -> np.ndarray:
        """To compute velocities do integration with the previous timestep.
        s(t+1) = s(t) + v(t) * dt
        v(t) = (s(t+1) - s(t)) / dt
        """
        speed = np.linalg.norm(velocities, axis=-1, keepdims=True)
        speed_sign = np.sign(
            velocities[..., 0, None] * np.cos(yaws)
            + velocities[..., 1, None] * np.sin(yaws)
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


def _perturb_pose(
    perturbation_se2: np.ndarray, batch: T.Dict[str, np.ndarray]
) -> T.Dict[str, np.ndarray]:
    """Perturb the pose of the agents in the batch"""
    perturbation_so2 = get_so2_from_se2(perturbation_se2)
    relative_yaw = get_yaw_from_se2(perturbation_se2)

    # Modify geometrical properties for agent features
    avails = batch["gt_states_avails"]
    positions = batch["gt_states"][avails, :2]
    directions = batch["gt_states"][avails, 5:7]
    batch["gt_states"][avails, :2] = transform_points(positions, perturbation_se2)
    batch["gt_states"][avails, 5:7] = transform_points(directions, perturbation_so2)
    batch["gt_states"][avails, 4] += relative_yaw

    # Modify geometrical properties for map features
    map_avails = batch["roadgraph_features_mask"]
    map_points = batch["roadgraph_features"][map_avails, :2]
    map_dirs = batch["roadgraph_features"][map_avails, 2:4]
    batch["roadgraph_features"][map_avails, :2] = transform_points(
        map_points, perturbation_se2
    )
    batch["roadgraph_features"][map_avails, 2:4] = transform_points(
        map_dirs, perturbation_so2
    )

    # Modify geometrical properties for tl features
    tl_avails = batch["tl_avails"]
    tl_points = batch["tl_states"][tl_avails, :2]
    batch["tl_states"][tl_avails, :2] = transform_points(tl_points, perturbation_se2)
    return batch


def move_frame_to_agent_of_idx(
    agent_idx: int, batch: T.Dict[str, np.ndarray]
) -> T.Dict[str, np.ndarray]:
    """Move the frame to the agent of index agent_idx.

    NOTE: batch will be mutated in place
    """
    centroid = batch["gt_states"][agent_idx, NUM_HISTORY_FRAMES, :2]
    yaw = batch["gt_states"][agent_idx, NUM_HISTORY_FRAMES, 4]

    transformation_matrix = get_transformation_matrix(centroid, yaw)
    return _perturb_pose(transformation_matrix, copy.deepcopy(batch))


def _generate_agent_centric_samples(
    sample: T.Dict[str, np.ndarray],
) -> T.Dict[str, np.ndarray]:
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


def _get_scenario_from_h5_file(
    file: T.Mapping[str, np.ndarray],
) -> T.Dict[str, np.ndarray]:
    return {
        "gt_states": np.array(file["gt_states"]).astype(
            np.float32
        ),  # [N_AGENTS, TIME, FEATS]
        "gt_states_avails": np.array(file["gt_states_avails"]).astype(
            np.bool_
        ),  # [N_AGENTS, TIME,]
        "actor_type": np.array(file["actor_type"]).astype(np.int64),  # [N_AGENTS,]
        "is_sdc": np.array(file["is_sdc"]).astype(np.bool_),  # [N_AGENTS,]
        "tracks_to_predict": np.array(file["tracks_to_predict"]).astype(
            np.bool_
        ),  # [N_AGENTS,]
        "roadgraph_features": np.array(file["roadgraph_features"]).astype(
            np.float32
        ),  # [N_POLYLINE, N_POINTS, FEATS]
        "roadgraph_features_mask": np.array(file["roadgraph_features_mask"]).astype(
            np.bool_
        ),  # [N_POLYLINE, N_POINTS]
        "roadgraph_features_types": np.array(file["roadgraph_features_types"]).astype(
            np.int64
        ),  # [N_POLYLINE,]
        "tl_states": np.array(file["tl_states"]).astype(
            np.float32
        ),  # [N_TRAFFIC_LIGHTS, TIME, 2]
        "tl_states_categorical": np.array(file["tl_states_categorical"]).astype(
            np.int64
        ),  # [N_TRAFFIC_LIGHTS, TIME,]
        "tl_avails": np.array(file["tl_avails"]).astype(
            np.bool_
        ),  # [N_TRAFFIC_LIGHTS, TIME,]
    }


class ScenarioDataset(Dataset):
    def __init__(
        self,
        filepath: str,
    ):
        self.datadir = pathlib.Path(filepath)

        with open(self.datadir / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def __len__(self) -> int:
        return len(self.metadata["scenario_idx_to_uuid"])

    def __getitem__(self, scenario_idx: int) -> T.List[T.Dict[str, np.ndarray]]:
        scenario_uuid = self.metadata["scenario_idx_to_uuid"][scenario_idx]
        agent_indices = self.metadata["scenario_uuid_to_actors"][scenario_uuid]

        with open(self.datadir / f"scenario_{scenario_uuid}.pkl", "rb") as f:
            sample = pickle.load(f)

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
            agent_sample["scenario_id"] = np.array(scenario_idx).astype(np.int64)
            batches.append(agent_sample)

        return batches


class AgentCentricDataset(Dataset):
    def __init__(
        self,
        filepath: str,
    ):
        self.datadir = pathlib.Path(filepath)
        with open(self.datadir / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def __len__(self) -> int:
        return len(self.metadata["coupled_indices"])

    def __getitem__(self, idx: int) -> T.Dict[str, np.ndarray]:
        scenario_idx, agent_idx = self.metadata["coupled_indices"][idx]
        scenario_uuid = self.metadata["scenario_idx_to_uuid"][scenario_idx]

        with open(self.datadir / f"scenario_{scenario_uuid}.pkl", "rb") as f:
            sample = pickle.load(f)

        # TODO: Maybe we should remove this function and just use the sample as is
        sample = _get_scenario_from_h5_file(sample)
        sample = move_frame_to_agent_of_idx(agent_idx, sample)

        # TODO: Move to feature generation
        sample["gt_features"], sample["gt_features_avails"] = _generate_agent_features(
            sample["gt_states"], sample["gt_states_avails"]
        )

        # TODO: Maybe move to data generation?
        sample["scenario_id"] = np.array(scenario_idx).astype(np.int64)
        sample["agent_to_predict"] = np.array(agent_idx).astype(np.int64)
        return sample
