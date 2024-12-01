from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Generic, TypeVar, Final
from collections import OrderedDict

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
from waymo_loader.feature_description import (
    STATE_FEATURES,
    ROADGRAPH_FEATURES,
    _ROADGRAPH_TYPE_TO_IDX
)

# Define generic type for arrays so that we can inspect when our features
# have numpy arrays or torch tensors
Array = TypeVar("Array", np.ndarray, Tensor)

MAX_NUM_AGENTS = 8
MAX_POLYLINE_LENGTH = 100


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


# def generate_roadgraph_features_dict(
#     roadgraph_features: RoadGraphSampleFeatures[np.ndarray],
# ) -> np.ndarray:

#     # TODO: Implement the traffic lights parsing
#     # if config.model_params.use_traffic_lights:
#     #     traffic_lights_features = utils.one_hot_numpy_adaptor(
#     #         dataset_output.roadgraph_features.roadgraph_tl_status, config.model_params.n_traffic_light_status_types
#     #     )
#     #     features.append(traffic_lights_features)

#     # The numerical values are not contiguous, we remap them to an ordered sequence of integers
#     mapping_fn = np.vectorize(lambda a: _ROADGRAPH_SAMPLE_TYPE_MAP_RESAMPLED.get(a, a))
#     ordinal_values = mapping_fn(roadgraph_features.map_elements_type_idx.astype(int))

#     # Numpy does not have a one hot encoding function, we use pytorch to do it.
#     n_types = len(list(_ROADGRAPH_SAMPLE_TYPE_MAP_RESAMPLED))
#     one_hot_encoded = F.one_hot(torch.from_numpy(ordinal_values), n_types).numpy()

#     features = [
#         roadgraph_features.roadgraph_xy,
#         roadgraph_features.roadgraph_dir,
#         one_hot_encoded,
#     ]
#     return np.concatenate(features, -1)


class WaymoDatasetHelper(object):

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

        extent = np.concatenate(
            [
                decoded_example["state/current/length"],
                decoded_example["state/current/width"],
                decoded_example["state/current/height"],
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
            extent=extent.astype(np.float32),
            actor_type=decoded_example["state/type"].astype(np.intp),
            is_sdc=decoded_example["state/is_sdc"].astype(np.bool_),
        )


ROADGRAPH_TYPES_OF_INTEREST: Final = {
    "LaneCenter-Freeway",
    "LaneCenter-SurfaceStreet",
    "LaneCenter-BikeLane",
    "RoadEdgeBoundary",
    "RoadEdgeMedian",
    "StopSign",
    "Crosswalk",
    "SpeedBump",
}

def get_bounding_box(polyline: np.ndarray) -> np.ndarray:
    """returns xmax, ymax, xmin, ymin of a polyline"""
    max_vals = np.max(polyline, axis=0)
    min_vals = np.min(polyline, axis=0)
    return np.concatenate([max_vals, min_vals], axis=1)
    
def get_bounds_from_points(xy_positions: np.ndarray, padding: float) -> np.ndarray:
    """ Gets the bounding box of that encloses all the points, (not the smallest one).
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

_MIN_NUM_OF_POINTS_IN_ROADGRAPH: Final = 100
_BOUNDS_PADDING_M: Final = 10.0

def _filter_inside_relevant_area(
    map_positions: np.ndarray, agent_positions: np.ndarray 
) -> Dict[str, np.ndarray]:
    """Filter data inside relevant area, returns a mask"""
    # If the map is already small then just return a valid mask
    if len(map_positions) <= _MIN_NUM_OF_POINTS_IN_ROADGRAPH:
        return np.ones_like(map_positions[..., 0]).astype(np.bool_)
    # Increase padding until we find minimum number of points
    current_padding_m = _BOUNDS_PADDING_M
    while True:
        bounds = get_bounds_from_points(agent_positions, current_padding_m)
        inside_bounds_mask = get_inside_bounds_mask(map_positions, bounds)
        n_inside_mask = np.sum(inside_bounds_mask)
        
        if n_inside_mask > _MIN_NUM_OF_POINTS_IN_ROADGRAPH:
            break
        current_padding_m += 10.0
    return inside_bounds_mask


def _parse_roadgraph_features(
    decoded_example: Dict[str, np.ndarray], to_ego_se3: np.ndarray, valid_positions: np.ndarray
) -> Dict[str, torch.Tensor]:
    # get only the rotation component to transform the polyline directions
    # rotation = get_so2_from_se2(to_ego_se3)
    points = transform_points(decoded_example["roadgraph_samples/xyz"][:, :2], to_ego_se3)
    
    ids = decoded_example["roadgraph_samples/id"][:, 0]
    valid = decoded_example["roadgraph_samples/valid"][:, 0].astype(np.bool_)
    types = decoded_example["roadgraph_samples/type"][:, 0].astype(np.int64)
    
    idx_of_iterest = [_ROADGRAPH_TYPE_TO_IDX[type] for type in ROADGRAPH_TYPES_OF_INTEREST]

    points = points[valid]  # [M, 2]
    ids = ids[valid].astype(np.int32) # [M,]
    mask_types = np.isin(types[valid], list(idx_of_iterest)) # [M,]
     
    map_mask = _filter_inside_relevant_area(points, valid_positions)
    total_mask = mask_types & map_mask
    
    valid_points = points[total_mask]
    valid_ids = ids[total_mask]

    unique_ids, _ = np.unique(valid_ids, return_counts=True)

    # Chop the polylines into smaller pieces to have minimum length
    chopped_polylines = []
    for id in unique_ids:
        polyline = valid_points[valid_ids == id]
        for i in range(0, len(polyline), MAX_POLYLINE_LENGTH):
            chopped_polylines.append(polyline[i : i + MAX_POLYLINE_LENGTH])
        
    masks = [np.ones((len(seq), 1), dtype=np.bool8) for seq in chopped_polylines]

    nested_tensor = torch.nested.nested_tensor(chopped_polylines, dtype=torch.float)
    padded_tensor = torch.nested.to_padded_tensor(nested_tensor, padding=0.0)

    nested_masks = torch.nested.nested_tensor(masks, dtype=torch.bool)
    padded_tensor_mask = torch.nested.to_padded_tensor(nested_masks, padding=False)
    return {
        "roadgraph_features": padded_tensor.numpy(),
        "roadgraph_features_mask": padded_tensor_mask.numpy(),
    }


def _generate_features(
    decoded_example: Dict[str, np.ndarray], future_num_frames=80, max_n_agents=MAX_NUM_AGENTS
) -> Dict[str, np.ndarray]:
    # The sdc is also considered so that we can retrieve its features
    tracks_to_predict = (decoded_example["state/tracks_to_predict"].squeeze() > 0).astype(np.bool_)
    is_sdc_mask = (decoded_example["state/is_sdc"].squeeze() > 0).astype(np.bool_)
    tracks_to_predict = np.logical_or(tracks_to_predict, is_sdc_mask)

    agent_features = WaymoDatasetHelper.generate_multi_agent_features(decoded_example)
    agent_features.crop_to_desired_prediction_horizon(future_num_frames)
    # Now remove the unwanted agents by filtering with tracks to predict
    agent_features.filter_with_mask(tracks_to_predict)

    # Transform the scene to have the ego vehicle at the origin
    ego_idx = np.argmax(agent_features.is_sdc.squeeze())
    to_sdc_se3 = get_transformation_matrix(
        agent_features.centroid[ego_idx],
        agent_features.yaw[ego_idx],
    )
    agent_features.transform_with_se3(to_sdc_se3)
    
    valid_history_positions = agent_features.history_positions[agent_features.history_availabilities]
    valid_target_positions = agent_features.target_positions[agent_features.target_availabilities]
    valid_positions = np.concatenate((valid_history_positions, valid_target_positions), axis=0)

    map_features = _parse_roadgraph_features(decoded_example, to_sdc_se3, valid_positions)

    # Now pad the remaining agents to the max number of agents, needed to have tensors the same size.
    # We also add 1 to the max number of agents to account for the ego vehicle
    agent_features.force_pad_batch_size(max_n_agents + 1)
    return agent_features.__dict__, map_features


def collate_waymo(payloads: List[Any]) -> Dict[str, Tensor]:
    # Maps should be collated differently (concatenated in the same dimension)
    batch = dict()
    batch.update(default_collate([value[0] for value in payloads]))

    batch["roadgraph_features"] = torch.nested.nested_tensor(
        [value[1]["roadgraph_features"] for value in payloads]
    )
    batch["roadgraph_features_mask"] = torch.nested.nested_tensor(
        [value[1]["roadgraph_features_mask"] for value in payloads]
    )
    return default_convert(batch)


def parse_concatenated_tensor(
    concatenated_tensor: np.ndarray, feature_dict: OrderedDict
) -> Dict[str, np.ndarray]:

    agent_length, _ = concatenated_tensor.shape

    parsed_features = {}
    start_idx = 0
    for feature_name, feature_descriptor in feature_dict.items():
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
            self.dataset_len = len(file["actor_merged_features"])

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # NOTE: We open the dataset here so that each worker has its own file handle
        if self.dataset is None:
            self.dataset = h5py.File(self.filepath, "r", libver="latest", swmr=True)

        actor_merged_features = np.array(self.dataset["actor_merged_features"][idx])
        roadgraph_merged_features = np.array(self.dataset["roadgraph_merged_features"][idx])

        data_fetched = {}
        data_fetched.update(parse_concatenated_tensor(actor_merged_features, STATE_FEATURES))
        data_fetched.update(
            parse_concatenated_tensor(roadgraph_merged_features, ROADGRAPH_FEATURES)
        )
        return _generate_features(data_fetched)

    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()
