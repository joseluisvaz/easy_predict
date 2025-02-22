import typing as T
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Final, Generic, List, Optional, TypeVar

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate, default_convert

from common_utils.geometry import get_so2_from_se2, get_transformation_matrix, get_yaw_from_se2, transform_points
from common_utils.tensor_utils import force_pad_batch_size
from data_utils.feature_description import (
    _ROADGRAPH_TYPE_TO_IDX,
    NUM_HISTORY_FRAMES,
    ROADGRAPH_FEATURES,
    STATE_FEATURES,
    SUBSAMPLE_SEQUENCE,
    TRAFFIC_LIGHT_FEATURES,
    MAP_MIN_NUM_OF_POINTS,
    MAP_BOUNDS_PADDING_M,
)

ROADGRAPH_TYPES_OF_INTEREST: Final = {
    "LaneCenter-Freeway",
    "LaneCenter-SurfaceStreet",
    "LaneCenter-BikeLane",
    "RoadLine-BrokenSingleWhite",
    "RoadLine-SolidSingleWhite",
    "RoadLine-SolidDoubleWhite",
    "RoadLine-BrokenSingleYellow",
    "RoadLine-BrokenDoubleYellow",
    "Roadline-SolidSingleYellow",
    "Roadline-SolidDoubleYellow",
    "RoadLine-PassingDoubleYellow",
    "RoadEdgeBoundary",
    "RoadEdgeMedian",
    "StopSign",
    "Crosswalk",
    "SpeedBump",
}

ROADGRAPH_TYPES_TO_SUBSAMPLE = [
    "LaneCenter-Freeway",
    "LaneCenter-SurfaceStreet",
    "LaneCenter-BikeLane",
    "RoadLine-BrokenSingleWhite",
    "RoadLine-SolidSingleWhite",
    "RoadLine-SolidDoubleWhite",
    "RoadLine-BrokenSingleYellow",
    "RoadLine-BrokenDoubleYellow",
    "Roadline-SolidSingleYellow",
    "Roadline-SolidDoubleYellow",
    "RoadLine-PassingDoubleYellow",
    "RoadEdgeBoundary",
    "RoadEdgeMedian",
    "Crosswalk",
]

MAX_NUM_POLYLINES: Final = 800
SUBSAMPLE_POLYLINE: T.Final = 4
MAX_POLYLINE_LENGTH: T.Final = 20


# Define generic type for arrays so that we can inspect when our features
# have numpy arrays or torch tensors
Array = TypeVar("Array", np.ndarray, Tensor)


@dataclass
class MultiAgentFeatures(Generic[Array]):
    # The order of the states is defined as in the WOMD metrics requirements for gt state ordering
    # x, y, length, width, bbox_yaw, velocity_x, velocity_y
    gt_states: Array
    gt_states_avails: Array
    actor_type: Array
    is_sdc: Array
    tracks_to_predict: Array

    def filter_with_mask(self, mask: Array) -> None:
        self.gt_states = self.gt_states[mask]
        self.gt_states_avails = self.gt_states_avails[mask]
        self.actor_type = self.actor_type[mask]
        self.is_sdc = self.is_sdc[mask]
        self.tracks_to_predict = self.tracks_to_predict[mask]

    def force_pad_batch_size(self, max_n_agents: int) -> None:
        self.gt_states = force_pad_batch_size(self.gt_states, max_n_agents)
        self.gt_states_avails = force_pad_batch_size(self.gt_states_avails, max_n_agents)
        self.actor_type = force_pad_batch_size(self.actor_type, max_n_agents)
        self.is_sdc = force_pad_batch_size(self.is_sdc, max_n_agents)
        self.tracks_to_predict = force_pad_batch_size(self.tracks_to_predict, max_n_agents)

    def transform_with_se3(self, transform: Array) -> None:
        rotation = get_so2_from_se2(transform)  # type: ignore
        relative_yaw = get_yaw_from_se2(transform)  # type: ignore

        avails = self.gt_states_avails

        gt_positions = self.gt_states[avails, :2]
        gt_velocities = self.gt_states[avails, 5:7]
        self.gt_states[avails, :2] = transform_points(gt_positions, transform)
        self.gt_states[avails, 5:7] = transform_points(gt_velocities, rotation)
        self.gt_states[avails, 4] += relative_yaw


class WaymoDatasetHelper(object):

    @staticmethod
    def generate_multi_agent_features(
        decoded_example: Dict[str, np.ndarray]
    ) -> MultiAgentFeatures[np.ndarray]:
        # The order of the states is defined as in the WOMD metrics requirements for gt state ordering
        def get_states(temporal_suffix: str) -> np.ndarray:
            """Returns the states for a specific temporal suffix."""
            return np.stack(
                [
                    decoded_example[f"state/{temporal_suffix}/x"],
                    decoded_example[f"state/{temporal_suffix}/y"],
                    decoded_example[f"state/{temporal_suffix}/length"],
                    decoded_example[f"state/{temporal_suffix}/width"],
                    decoded_example[f"state/{temporal_suffix}/bbox_yaw"],
                    decoded_example[f"state/{temporal_suffix}/velocity_x"],
                    decoded_example[f"state/{temporal_suffix}/velocity_y"],
                ],
                axis=-1,
            )

        past_states = get_states("past")
        cur_states = get_states("current")
        future_states = get_states("future")

        gt_states = np.concatenate([past_states, cur_states, future_states], axis=1)
        gt_states_avails = np.concatenate(
            [
                decoded_example["state/past/valid"] > 0,
                decoded_example["state/current/valid"] > 0,
                decoded_example["state/future/valid"] > 0,
            ],
            axis=1,
        ).astype(np.bool_)

        gt_states = gt_states[:, ::SUBSAMPLE_SEQUENCE]
        gt_states_avails = gt_states_avails[:, ::SUBSAMPLE_SEQUENCE]

        # By convention velocities have one element less
        return MultiAgentFeatures(
            gt_states=gt_states.astype(np.float32),
            gt_states_avails=gt_states_avails.astype(np.bool_),
            actor_type=decoded_example["state/type"].squeeze().astype(np.intp),
            is_sdc=(decoded_example["state/is_sdc"] > 0).squeeze().astype(np.bool_),
            tracks_to_predict=(decoded_example["state/tracks_to_predict"] > 0)
            .squeeze()
            .astype(np.bool_),
        )


def pad_or_trim_first_dimension(tensor: np.array, value: Any) -> np.ndarray:
    """Pads or trims an array to a specific size."""

    ndim = tensor.ndim
    assert ndim > 0

    if tensor.shape[0] < MAX_NUM_POLYLINES:
        padding = MAX_NUM_POLYLINES - tensor.shape[0]
        padding_sequence = [(0, padding)] + [(0, 0)] * (ndim - 1)
        padded_tensor = np.pad(tensor, padding_sequence, mode="constant", constant_values=value)
        return padded_tensor
    warnings.warn("Trimming the tensor to the maximum number of polylines")
    return tensor[:MAX_NUM_POLYLINES]


def get_bounding_box(polyline: np.ndarray) -> np.ndarray:
    """returns xmax, ymax, xmin, ymin of a polyline"""
    max_vals = np.max(polyline, axis=0)
    min_vals = np.min(polyline, axis=0)
    return np.concatenate([max_vals, min_vals], axis=1)


def get_bounds_from_points(xy_positions: np.ndarray, padding: float) -> np.ndarray:
    """Gets the bounding box of that encloses all the points, (not the smallest one).
        Bounding box described by lower left and upper right corner
    Args:
        xy_positions: xy positions
        padding: padding to add in max and min of the bounding box
    Returns:
        lower left and upper right corners
    """
    assert xy_positions.ndim == 2, xy_positions.shape[1] == 2
    lower_left_approximation = np.min(xy_positions, axis=0) - padding
    upper_right_approximation = np.max(xy_positions, axis=0) + padding
    return np.stack([lower_left_approximation, upper_right_approximation], axis=0)


def get_inside_bounds_mask(xy_positions: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Gets elements within bounds.
    Args:
        xy_positions (np.ndarray): XY of the center
        bounds (np.ndarray): array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]
    Returns:
        np.ndarray: mask of elements inside bound
    """
    assert xy_positions.ndim == 2 and xy_positions.shape[1] == 2
    assert bounds.ndim == 2 and bounds.shape == (2, 2)

    x_center = xy_positions[:, 0]
    y_center = xy_positions[:, 1]

    x_min_in = x_center > bounds[0, 0]
    y_min_in = y_center > bounds[0, 1]
    x_max_in = x_center < bounds[1, 0]
    y_max_in = y_center < bounds[1, 1]
    return x_min_in & y_min_in & x_max_in & y_max_in


def _filter_inside_relevant_area(
    map_positions: np.ndarray, agent_positions: np.ndarray
) -> Dict[str, np.ndarray]:
    """Filter data inside relevant area, returns a mask"""
    # If the map is already small then just return a valid mask
    if len(map_positions) <= MAP_MIN_NUM_OF_POINTS:
        return np.ones_like(map_positions[..., 0]).astype(np.bool_)
    # Increase padding until we find minimum number of points
    current_padding_m = MAP_BOUNDS_PADDING_M
    while True:
        bounds = get_bounds_from_points(agent_positions, current_padding_m)
        inside_bounds_mask = get_inside_bounds_mask(map_positions, bounds)
        n_inside_mask = np.sum(inside_bounds_mask)

        if n_inside_mask > MAP_MIN_NUM_OF_POINTS:
            break
        current_padding_m += 10.0
    return inside_bounds_mask


def _parse_roadgraph_features(
    decoded_example: Dict[str, np.ndarray], to_ego_se3: np.ndarray, valid_positions: np.ndarray
) -> Dict[str, torch.Tensor]:

    def _apply_validity_masks(points, dirs, valid, types, ids):
        """Filter the roadgraph based on validity and type of the roadgraph element"""
        points = points[valid]  # [M, 2]
        dirs = dirs[valid]  # [M, 2]
        ids = ids[valid].astype(np.int32)  # [M,]
        types = types[valid]  # [M,]

        idx_of_iterest = [_ROADGRAPH_TYPE_TO_IDX[type] for type in ROADGRAPH_TYPES_OF_INTEREST]
        mask_types = np.isin(types, list(idx_of_iterest))  # [M,]
        map_mask = _filter_inside_relevant_area(points, valid_positions)
        total_mask = mask_types & map_mask

        valid_points = points[total_mask]
        valid_dirs = dirs[total_mask]
        valid_ids = ids[total_mask]
        valid_types = types[total_mask]
        return valid_points, valid_dirs, valid_ids, valid_types

    # get only the rotation component to transform the polyline directions
    rotation = get_so2_from_se2(to_ego_se3)
    points = transform_points(decoded_example["roadgraph_samples/xyz"][:, :2], to_ego_se3)
    dirs = transform_points(decoded_example["roadgraph_samples/dir"][:, :2], rotation)
    ids = decoded_example["roadgraph_samples/id"][:, 0]
    valid = decoded_example["roadgraph_samples/valid"][:, 0].astype(np.bool_)
    types = decoded_example["roadgraph_samples/type"][:, 0].astype(np.int64)

    valid_points, valid_dirs, valid_ids, valid_types = _apply_validity_masks(
        points, dirs, valid, types, ids
    )
    unique_ids, _ = np.unique(valid_ids, return_counts=True)

    def _subsample_sequence(sequence: np.ndarray, subsample: int):
        if len(sequence) <= 3:
            return sequence
        indices = np.arange(1, len(sequence) - 1, subsample)
        indices = np.concatenate(([0], indices, [len(sequence) - 1]))
        return sequence[indices]

    ROADGRAPH_IDX_TO_SUBSAMPLE = {
        _ROADGRAPH_TYPE_TO_IDX[_type] for _type in ROADGRAPH_TYPES_TO_SUBSAMPLE
    }

    def _select_and_decompose_sequences(flattened_sequences: np.ndarray, types: np.ndarray):
        """Select the unique ids and decompose the polylines into smaller pieces"""
        decomposed = []
        for id in unique_ids:
            indices = valid_ids == id
            polyline_type = types[indices][0]  # Get the type of this polyline
            subsample_factor = (
                SUBSAMPLE_POLYLINE if polyline_type in ROADGRAPH_IDX_TO_SUBSAMPLE else 1
            )

            subsequence = _subsample_sequence(flattened_sequences[indices], subsample_factor)
            for i in range(0, len(subsequence), MAX_POLYLINE_LENGTH):
                decomposed.append(subsequence[i : i + MAX_POLYLINE_LENGTH])
        return decomposed

    def _resample_ids(flattened_ids: np.ndarray, types: np.ndarray):
        """Similar to the previous function but has a counter to populate the new ids"""
        decomposed = []
        id_counter = 1
        for id in unique_ids:
            indices = valid_ids == id
            polyline_type = types[indices][0]  # Get the type of this polyline
            subsample_factor = (
                SUBSAMPLE_POLYLINE if polyline_type in ROADGRAPH_IDX_TO_SUBSAMPLE else 1
            )

            subsequence = _subsample_sequence(flattened_ids[indices], subsample_factor)
            for i in range(0, len(subsequence), MAX_POLYLINE_LENGTH):
                decomposed.append(
                    id_counter
                    * np.ones_like(subsequence[i : i + MAX_POLYLINE_LENGTH]).astype(np.int16)
                )
                id_counter += 1
        return decomposed

    chopped_polylines = _select_and_decompose_sequences(valid_points, valid_types)
    chopped_dirs = _select_and_decompose_sequences(valid_dirs, valid_types)
    chopped_types = _select_and_decompose_sequences(valid_types, valid_types)
    chopped_ids = _resample_ids(valid_ids, valid_types)
    masks = [np.ones((len(seq), 1), dtype=np.bool8) for seq in chopped_polylines]

    def _nest_and_pad(
        tensor_sequence: List[np.ndarray], torch_type: Any, padding: Any
    ) -> np.ndarray:
        # Create a max size sensor to make the padding valid, we can remove it at the end
        dummy_tensor = torch.zeros(
            (MAX_POLYLINE_LENGTH, *tensor_sequence[0].shape[1:]), dtype=torch_type
        )
        tensor_sequence.append(dummy_tensor)

        nested_tensor = torch.nested.nested_tensor(tensor_sequence, dtype=torch_type)
        padded_tensor = torch.nested.to_padded_tensor(nested_tensor, padding=padding).numpy()

        # Remove the dummy tensor
        return padded_tensor[:-1]

    chopped_polylines = _nest_and_pad(chopped_polylines, torch.float, 0.0)
    chopped_dirs = _nest_and_pad(chopped_dirs, torch.float, 0.0)
    features = np.concatenate((chopped_polylines, chopped_dirs), axis=-1)
    masks = _nest_and_pad(masks, torch.bool, False)
    chopped_types = _nest_and_pad(chopped_types, torch.int64, 0)
    chopped_ids = _nest_and_pad(chopped_ids, torch.int16, 0)

    return {
        "roadgraph_features": pad_or_trim_first_dimension(
            features, 0.0
        ),  # [N_POLYLINE, N_POINTS, FEATS]
        "roadgraph_features_mask": pad_or_trim_first_dimension(
            masks, False
        ).squeeze(),  # [N_POLYLINE, N_POINTS]
        "roadgraph_features_types": pad_or_trim_first_dimension(chopped_types, 0)[
            :, 0
        ],  # [N_POLYLINE,] # Only one type per polyline
        "roadgraph_features_ids": pad_or_trim_first_dimension(chopped_ids, 0)[
            :, 0
        ],  # [N_POLYLINE,] # Only one id per polyline
    }


def _parse_traffic_light_features(
    decoded_example: Dict[str, np.ndarray], to_agent_se3: np.ndarray
) -> Dict[str, np.ndarray]:
    past_tl_pos = np.stack(
        (
            decoded_example["traffic_light_state/past/x"],
            decoded_example["traffic_light_state/past/y"],
        ),
        axis=-1,
    )
    current_tl_pos = np.stack(
        (
            decoded_example["traffic_light_state/current/x"],
            decoded_example["traffic_light_state/current/y"],
        ),
        axis=-1,
    )
    tl_pos = np.concatenate((past_tl_pos, current_tl_pos), axis=1)

    tl_states_categorical = np.concatenate(
        (
            decoded_example["traffic_light_state/past/state"],
            decoded_example["traffic_light_state/current/state"],
        ),
        axis=1,
    )
    tl_avails = np.concatenate(
        (
            decoded_example["traffic_light_state/past/valid"],
            decoded_example["traffic_light_state/current/valid"],
        ),
        axis=1,
    )

    tl_pos = tl_pos[:, ::SUBSAMPLE_SEQUENCE]
    tl_states_categorical = tl_states_categorical[:, ::SUBSAMPLE_SEQUENCE]
    tl_avails = tl_avails[:, ::SUBSAMPLE_SEQUENCE].astype(np.bool_)

    # Invalid values are set as -1 so get rid of them by multiplying by 0
    tl_pos *= tl_avails[..., None].astype(np.int16)
    tl_states_categorical *= tl_avails.astype(np.int16)

    tl_pos[tl_avails] = transform_points(tl_pos[tl_avails], to_agent_se3)
    assert len(tl_pos.shape) == 3, "tmp assert"
    assert tl_states_categorical.shape[1] == 6, "tmp assert"

    return {
        "tl_states": tl_pos,  # [N_TRAFFIC_LIGHTS, TIME, 2]
        "tl_states_categorical": tl_states_categorical,  # [N_TRAFFIC_LIGHTS, TIME]
        "tl_avails": tl_avails,  # [N_TRAFFIC_LIGHTS, TIME]
    }


def _generate_features(
    decoded_example: Dict[str, np.ndarray], train_with_tracks_to_predict: bool = False
) -> Dict[str, np.ndarray]:
    agent_features = WaymoDatasetHelper.generate_multi_agent_features(decoded_example)

    # # Now remove the unwanted agents by filtering with tracks to predict
    agent_features_current_state = agent_features.gt_states[:, NUM_HISTORY_FRAMES]

    # Transform the scene to have the ego vehicle at the origin
    agent_idx = np.argmax(agent_features.is_sdc.squeeze())
    to_agent_se3 = get_transformation_matrix(
        agent_features_current_state[agent_idx, :2],  # x, y
        agent_features_current_state[agent_idx, 4],  # yaw
    )
    agent_features.transform_with_se3(to_agent_se3)

    valid_positions = agent_features.gt_states[agent_features.gt_states_avails][:, :2]
    map_features = _parse_roadgraph_features(decoded_example, to_agent_se3, valid_positions)
    tl_features = _parse_traffic_light_features(decoded_example, to_agent_se3)

    if train_with_tracks_to_predict:
        MAX_PREDICATABLE_AGENTS: Final = 9  # 8 + ego
        agent_features.force_pad_batch_size(MAX_PREDICATABLE_AGENTS)
        
    current_avails = agent_features.gt_states_avails[:, NUM_HISTORY_FRAMES].astype(np.bool_)
    tracks_to_predict = agent_features.tracks_to_predict.astype(np.bool_)
    assert np.all(current_avails[tracks_to_predict])
    

    return {
        "gt_states": agent_features.gt_states.astype(
            np.float32
        ),  # [N_AGENTS, TIME, FEATS=7] (x, y, length, width, yaw, velocity_x, velocity_y)
        "gt_states_avails": agent_features.gt_states_avails.astype(np.bool_),  # [N_AGENTS, TIME]
        "actor_type": agent_features.actor_type.astype(np.int64),  # [N_AGENTS,]
        "is_sdc": agent_features.is_sdc.astype(np.bool_),  # [N_AGENTS,]
        "tracks_to_predict": agent_features.tracks_to_predict.astype(np.bool_),  # [N_AGENTS,]
        "roadgraph_features": map_features["roadgraph_features"].astype(
            np.float32
        ),  # [N_POLYLINE, N_POINTS, FEATS=4] (x, y, dx, dy)
        "roadgraph_features_mask": map_features["roadgraph_features_mask"].astype(
            np.bool_
        ),  # [N_POLYLINE, N_POINTS]
        "roadgraph_features_types": map_features["roadgraph_features_types"].astype(
            np.int64
        ),  # [N_POLYLINE,]
        "roadgraph_features_ids": map_features["roadgraph_features_ids"].astype(
            np.int16
        ),  # [N_POLYLINE,]
        "tl_states": tl_features["tl_states"].astype(np.float32),  # [N_TRAFFIC_LIGHTS, TIME, 2]
        "tl_states_categorical": tl_features["tl_states_categorical"].astype(
            np.int64
        ),  # [N_TRAFFIC_LIGHTS, TIME]
        "tl_avails": tl_features["tl_avails"].astype(np.bool_),  # [N_TRAFFIC_LIGHTS, TIME]
    }


def collate_waymo_concatenate(payloads: List[Any]) -> Dict[str, Tensor]:
    collated_batch = {}
    for key in payloads[0]:
        collated_batch[key] = np.concatenate([payload[key] for payload in payloads])
    return default_convert(collated_batch)

def collate_waymo_stack(payloads: List[Any]) -> Dict[str, Tensor]:
    return default_convert(default_collate(payloads))

def collate_waymo_scenario(payloads: T.List[T.List[T.Dict[str, np.ndarray]]]) -> Dict[str, Tensor]:
    
    # First collation will do stacking of the tensors
    collated_subbatches = []
    for subbatch in payloads:
        collated_subbatches.append(default_collate(subbatch))
    
    # Second collation will do concatenation of the tensors
    collated_batch = {}
    for key in collated_subbatches[0]:
        collated_batch[key] = torch.concatenate([subbatch[key] for subbatch in collated_subbatches])
    
    return default_convert(collated_batch)

def parse_concatenated_tensor(
    concatenated_tensor: np.ndarray, feature_dict: OrderedDict, batch_first: bool = True
) -> Dict[str, np.ndarray]:
    agent_length, _ = concatenated_tensor.shape

    parsed_features = {}
    start_idx = 0
    for feature_name, feature_descriptor in feature_dict.items():
        shape = feature_descriptor.shape
        assert len(shape) <= 2, "feature shape should be at most 2D"

        if not batch_first:
            # e.g. traffic lights have the collection dimension as the second dim. but we
            # transpose it for encoding.
            shape = shape[::-1]

        feature_length = shape[1] if len(shape) > 1 else 1

        end_idx = start_idx + feature_length
        parsed_features[feature_name] = concatenated_tensor[:, start_idx:end_idx].reshape(
            agent_length, feature_length
        )
        start_idx = end_idx

    return parsed_features


class WaymoH5Dataset(Dataset):
    def __init__(self, filepath: str, train_with_tracks_to_predict: bool):
        """Do not open the h5 file here"""
        self.dataset: Optional[h5py.File] = None
        self.feature_description = STATE_FEATURES
        self.filepath = filepath
        self.train_with_tracks_to_predict = train_with_tracks_to_predict

        # Open and close dataset just to extract the length
        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as file:
            self.dataset_len = len(file["actor_merged_features"])

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # NOTE: We open the dataset here so that each worker has its own file handle
        if self.dataset is None:
            self.dataset = h5py.File(self.filepath, "r", libver="latest", swmr=True)

        actor_merged_features = np.array(self.dataset["actor_merged_features"][idx])
        roadgraph_merged_features = np.array(self.dataset["roadgraph_merged_features"][idx])
        tl_merged_features = np.array(self.dataset["tl_merged_features"][idx])

        data_fetched = {}
        data_fetched.update(parse_concatenated_tensor(actor_merged_features, STATE_FEATURES))
        data_fetched.update(
            parse_concatenated_tensor(roadgraph_merged_features, ROADGRAPH_FEATURES)
        )
        data_fetched.update(
            parse_concatenated_tensor(tl_merged_features, TRAFFIC_LIGHT_FEATURES, batch_first=False)
        )
        batch =  _generate_features(
            data_fetched, train_with_tracks_to_predict=self.train_with_tracks_to_predict
        )
        
        # Attach the index of the scenario for quick reference
        batch["scenario_id"] = np.array(idx).astype(np.int64)
        return batch
