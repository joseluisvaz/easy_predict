import copy
import pathlib
import pickle
import typing as T

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate, default_convert

from data_utils.feature_description import NUM_HISTORY_FRAMES
from utils.geometry import (
    get_so2_from_se2,
    get_transformation_matrix,
    get_yaw_from_se2,
    transform_points,
)


def collate_waymo_concatenate(
    payloads: T.List[T.Dict[str, np.ndarray]],
) -> T.Dict[str, np.ndarray]:
    collated_batch = {}
    for key in payloads[0]:
        collated_batch[key] = np.concatenate([payload[key] for payload in payloads])
    return default_convert(collated_batch)


def collate_waymo_stack(
    payloads: T.List[T.Dict[str, np.ndarray]],
) -> T.Dict[str, np.ndarray]:
    return default_convert(default_collate(payloads))


def collate_waymo_scenario(
    payloads: T.List[T.List[T.Dict[str, np.ndarray]]],
) -> T.Dict[str, np.ndarray]:
    # First collation will do stacking of the tensors
    collated_subbatches = []
    for subbatch in payloads:
        collated_subbatches.append(default_collate(subbatch))

    # Second collation will do concatenation of the tensors
    collated_batch = {}
    for key in collated_subbatches[0]:
        collated_batch[key] = np.concatenate(
            [subbatch[key] for subbatch in collated_subbatches]
        )

    return default_convert(collated_batch)


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

        sample = move_frame_to_agent_of_idx(agent_idx, sample)

        # TODO: Move to feature generation
        sample["gt_features"], sample["gt_features_avails"] = _generate_agent_features(
            sample["gt_states"], sample["gt_states_avails"]
        )

        # TODO: Maybe move to data generation?
        sample["scenario_id"] = np.array(scenario_idx).astype(np.int64)
        sample["agent_to_predict"] = np.array(agent_idx).astype(np.int64)
        return sample
