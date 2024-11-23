from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Generic, TypeVar

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate, default_convert

from common_utils.geometry import (
    transform_points,
    get_transformation_matrix,
    get_so2_from_se2,
    get_yaw_from_se2,
)
from common_utils.tensor_utils import force_pad_batch_size
from waymo_loader.feature_description import get_feature_description, STATE_FEATURES

# Define generic type for arrays so that we can inspect when our features
# have numpy arrays or torch tensors
Array = TypeVar("Array", np.ndarray, Tensor)

MAX_NUM_AGENTS = 8


@dataclass
class MultiAgentFeatures(Generic[Array]):
    # These features are centered around a specific agent,
    # these also go backwards in time
    history_positions: Array
    history_velocities: Array
    history_availabilities: Array
    history_yaws: Array

    target_positions: Array
    target_velocities: Array
    target_availabilities: Array
    target_yaws: Array

    # These are in an arbitrary frame they should not be transformed into a separate frame
    centroid: Array
    yaw: Array
    speed: Array

    extent: Array  # Extent of the object (bounding box)
    actor_type: Array
    is_sdc: Array

    def filter_with_mask(self, mask: Array) -> None:
        self.history_velocities = self.history_velocities[mask]
        self.history_positions = self.history_positions[mask]
        self.history_yaws = self.history_yaws[mask]
        self.history_availabilities = self.history_availabilities[mask]

        self.target_velocities = self.target_velocities[mask]
        self.target_positions = self.target_positions[mask]
        self.target_yaws = self.target_yaws[mask]
        self.target_availabilities = self.target_availabilities[mask]

        self.centroid = self.centroid[mask]
        self.yaw = self.yaw[mask]
        self.speed = self.speed[mask]
        self.extent = self.extent[mask]
        self.actor_type = self.actor_type[mask]
        self.is_sdc = self.is_sdc[mask]

    def force_pad_batch_size(self, max_n_agents: int) -> None:
        self.history_velocities = force_pad_batch_size(self.history_velocities, max_n_agents)
        self.history_positions = force_pad_batch_size(self.history_positions, max_n_agents)
        self.history_yaws = force_pad_batch_size(self.history_yaws, max_n_agents)
        self.history_availabilities = force_pad_batch_size(
            self.history_availabilities, max_n_agents
        )

        self.target_velocities = force_pad_batch_size(self.target_velocities, max_n_agents)
        self.target_positions = force_pad_batch_size(self.target_positions, max_n_agents)
        self.target_yaws = force_pad_batch_size(self.target_yaws, max_n_agents)
        self.target_availabilities = force_pad_batch_size(self.target_availabilities, max_n_agents)

        self.centroid = force_pad_batch_size(self.centroid, max_n_agents)
        self.yaw = force_pad_batch_size(self.yaw, max_n_agents)
        self.speed = force_pad_batch_size(self.speed, max_n_agents)
        self.extent = force_pad_batch_size(self.extent, max_n_agents)
        self.actor_type = force_pad_batch_size(self.actor_type, max_n_agents)
        self.is_sdc = force_pad_batch_size(self.is_sdc, max_n_agents)

    def transform_with_se3(self, transform: Array) -> None:
        rotation = get_so2_from_se2(transform)  # type: ignore

        h_avails = self.history_availabilities
        t_avails = self.target_availabilities

        self.history_positions[h_avails] = transform_points(
            self.history_positions[h_avails], transform
        )
        self.target_positions[t_avails] = transform_points(
            self.target_positions[t_avails], transform
        )
        self.target_velocities[t_avails] = transform_points(
            self.target_velocities[t_avails], rotation
        )
        self.history_velocities[h_avails] = transform_points(
            self.history_velocities[h_avails], rotation
        )

        # Adjust yaws accordingly
        relative_yaw = get_yaw_from_se2(transform)  # type: ignore
        self.history_yaws[h_avails] += relative_yaw
        self.target_yaws[t_avails] += relative_yaw

    def crop_to_desired_prediction_horizon(self, desired_length: int) -> None:
        """Crops the features to a desired length.

        Args:
            desired_length: config containing the desired length
        """
        maximum_length = self.target_availabilities.shape[1]

        if desired_length > maximum_length:
            raise ValueError("Desired length exceeded maximum length of the dataset")

        self.target_positions = self.target_positions[:, :desired_length]
        self.target_velocities = self.target_velocities[:, :desired_length]
        self.target_yaws = self.target_yaws[:, :desired_length]
        self.target_availabilities = self.target_availabilities[:, :desired_length]


DatasetOutputDicts = Dict[str, Array]


class WaymoDatasetHelper(object):
    @staticmethod
    def get_extent(decoded_example: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                decoded_example["state/current/length"],
                decoded_example["state/current/width"],
                decoded_example["state/current/height"],
            ],
            axis=-1,
        )

    @staticmethod
    def generate_multi_agent_features(
        decoded_example: Dict[str, np.ndarray]
    ) -> MultiAgentFeatures[np.ndarray]:
        past_states = np.stack(
            [
                decoded_example["state/past/x"],
                decoded_example["state/past/y"],
                decoded_example["state/past/bbox_yaw"],
                decoded_example["state/past/velocity_x"],
                decoded_example["state/past/velocity_y"],
                decoded_example["state/past/speed"],
            ],
            axis=-1,
        )

        cur_states = np.stack(
            [
                decoded_example["state/current/x"],
                decoded_example["state/current/y"],
                decoded_example["state/current/bbox_yaw"],
                decoded_example["state/current/velocity_x"],
                decoded_example["state/current/velocity_y"],
                decoded_example["state/current/speed"],
            ],
            axis=-1,
        )

        future_states = np.stack(
            [
                decoded_example["state/future/x"],
                decoded_example["state/future/y"],
                decoded_example["state/future/bbox_yaw"],
                decoded_example["state/future/velocity_x"],
                decoded_example["state/future/velocity_y"],
                decoded_example["state/future/speed"],
            ],
            axis=-1,
        )

        history_states = np.concatenate([past_states, cur_states], axis=1)
        history_availabilities = np.concatenate(
            [decoded_example["state/past/valid"] > 0, decoded_example["state/current/valid"] > 0],
            axis=1,
        ).astype(np.bool_)
        future_availabilities = decoded_example["state/future/valid"] > 0

        # By convention velocities have one element less
        return MultiAgentFeatures(
            history_positions=history_states[..., :2].astype(np.float32),
            history_velocities=history_states[..., 3:5].astype(np.float32),
            history_yaws=history_states[..., 2, None].astype(np.float32),
            history_availabilities=history_availabilities.astype(np.bool_),
            target_positions=future_states[..., :2].astype(np.float32),
            target_velocities=future_states[..., 3:5].astype(np.float32),
            target_yaws=future_states[..., 2, None].astype(np.float32),
            target_availabilities=future_availabilities.astype(np.bool_),
            centroid=cur_states[..., :2].astype(np.float32),
            yaw=cur_states[..., 2, None].astype(np.float32),
            speed=cur_states[..., 5, None].astype(np.float32),
            extent=WaymoDatasetHelper.get_extent(decoded_example).astype(np.float32),
            actor_type=decoded_example["state/type"].astype(np.intp),
            is_sdc=decoded_example["state/is_sdc"].astype(np.bool_),
        )


def _generate_features(
    decoded_example: Dict[str, np.ndarray], future_num_frames=80, max_n_agents=MAX_NUM_AGENTS
) -> Dict[str, np.ndarray]:
    # If a sample was not seen at all in the past, we declare the sample as invalid.
    tracks_to_predict = (decoded_example["state/tracks_to_predict"].squeeze() > 0).astype(np.bool_)
    is_sdc_mask = (decoded_example["state/is_sdc"].squeeze() > 0).astype(np.bool_)
    # The sdc is also considered so that we can retrieve its features
    tracks_to_predict = np.logical_or(tracks_to_predict, is_sdc_mask)

    multi_agent_features = WaymoDatasetHelper.generate_multi_agent_features(decoded_example)
    multi_agent_features.crop_to_desired_prediction_horizon(future_num_frames)

    # Get the index of the self driving car to transform to an ego-centric scene,
    # do this before removing the unwanted agents because it could be that we do not
    # want to predict ego
    ego_idx = np.argmax(multi_agent_features.is_sdc.squeeze())
    ego_centroid = multi_agent_features.centroid[ego_idx]
    ego_yaw = multi_agent_features.yaw[ego_idx]

    # Now remove the unwanted agents by filtering with tracks to predict
    multi_agent_features.filter_with_mask(tracks_to_predict)

    # From the surviving agents choose the firs one's centroid and yaw to center the scene
    to_sdc_se3 = get_transformation_matrix(
        ego_centroid,
        ego_yaw,
    )
    multi_agent_features.transform_with_se3(to_sdc_se3)

    # Now pad the remaining agents to the max number of agents, needed to have tensors the same size.
    # We also add 1 to the max number of agents to account for the ego vehicle
    multi_agent_features.force_pad_batch_size(max_n_agents + 1)

    return multi_agent_features.__dict__


def collate_waymo(payloads: List[Any]) -> Dict[str, Tensor]:
    # Collate the normal features like normal tensors, and convert it to pytorch
    batch = default_collate(payloads)
    torch_batch = default_convert(batch)
    return torch_batch


def parse_concatenated_tensor(concatenated_tensor: np.ndarray):

    agent_length, _ = concatenated_tensor.shape

    parsed_features = {}
    start_idx = 0
    for feature_name, feature_descriptor in STATE_FEATURES.items():
        shape = feature_descriptor.shape
        feature_length = shape[1] if len(shape) > 1 else 1

        end_idx = start_idx + feature_length
        parsed_features[feature_name] = concatenated_tensor[:, start_idx:end_idx].reshape(
            agent_length, feature_length
        )
        start_idx = end_idx

    return parsed_features


class WaymoH5Dataset(Dataset):
    def __init__(self, filepath: str):
        """Do not open the h5 file here"""
        self.dataset: Optional[h5py.File] = None
        self.feature_description = STATE_FEATURES
        self.filepath = filepath

        # Open and close dataset just to extract the length
        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as file:
            self.dataset_len = len(file["merged_features"])

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # NOTE: We open the dataset here so that each worker has its own file handle
        if self.dataset is None:
            self.dataset = h5py.File(self.filepath, "r", libver="latest", swmr=True)

        merged_features = np.array(self.dataset["merged_features"][idx])
        data_fetched = parse_concatenated_tensor(merged_features)

        features = _generate_features(data_fetched)
        return features

    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()
