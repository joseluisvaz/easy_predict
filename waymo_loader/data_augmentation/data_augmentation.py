import typing as T
from abc import ABC, abstractmethod

import numpy as np

from common_utils.geometry import get_so2_from_se2, get_transformation_matrix, get_yaw_from_se2, transform_points

# from utils.geometry import get_transformation_matrix
from waymo_loader.feature_description import NUM_HISTORY_FRAMES


def perturb_pose(perturbation_se2, batch: T.Dict[str, np.ndarray]):
    """Perturb the pose of the agents in the batch"""
    perturbation_so2 = get_so2_from_se2(perturbation_se2)
    relative_yaw = get_yaw_from_se2(perturbation_se2)

    # Modify geometrical properties for agent features
    avails = batch["gt_states_avails"]
    positions = batch["gt_states"][avails, :2]
    velocities = batch["gt_states"][avails, 5:7]
    batch["gt_states"][avails, :2] = transform_points(positions, perturbation_se2)
    batch["gt_states"][avails, 5:7] = transform_points(velocities, perturbation_so2)
    batch["gt_states"][avails, 4] += relative_yaw

    # Modify geometrical properties for map features
    map_avails = batch["roadgraph_features_mask"]
    map_points = batch["roadgraph_features"][map_avails, :2]
    map_dirs = batch["roadgraph_features"][map_avails, 2:4]
    batch["roadgraph_features"][map_avails, :2] = transform_points(map_points, perturbation_se2)
    batch["roadgraph_features"][map_avails, 2:4] = transform_points(map_dirs, perturbation_so2)
    return batch


class DataAugmentation(ABC):
    """Data perturbation abstract class.
    To be implemented by all data augmentation classes, each data
    augmentation class needs to decide which samples to augment
    """

    @abstractmethod
    def perturb(self, data: T.Dict[str, np.ndarray]) -> None:
        """Peturb the data sample
        Args:
            data (T.Dict[str, np.ndarray]): Data sample to be perturbed.
        """
        pass


class ComposedAugmentation(DataAugmentation):
    def __init__(self, perturbations: T.Sequence[DataAugmentation]) -> None:
        self.perturbations = perturbations

    def perturb(self, data: T.Dict[str, np.ndarray]) -> None:
        for perturbation in self.perturbations:
            perturbation.perturb(data)


class AnchorFrameAugmentation(DataAugmentation):
    def __init__(self, perturb_prob: float) -> None:
        assert perturb_prob >= 0.0

        self.perturb_prob = perturb_prob

    def perturb(self, data: T.Dict[str, np.ndarray]) -> None:
        """Transform the multiagent sample to a randomly sampled agent frame
        Args:
            data: Dict with a multiagent sample for predictions, everything should be in the ego frame.
        Returns:
            multiagent_sample transformed by the randomly sampled transformation
        """
        if np.random.rand() >= self.perturb_prob:
            return

        # Get valid agents, and check if its a valid track to predict
        current_availabilities = data["gt_states_avails"][:, NUM_HISTORY_FRAMES]
        tracks_to_predict = data["tracks_to_predict"]

        available_and_predictable = np.logical_and(current_availabilities, tracks_to_predict)
        agent_indices = np.arange(len(available_and_predictable))
        selected_agent_id = np.random.choice(agent_indices[available_and_predictable])

        centroid = data["gt_states"][selected_agent_id, NUM_HISTORY_FRAMES, :2]
        yaw = data["gt_states"][selected_agent_id, NUM_HISTORY_FRAMES, 4]

        transformation_matrix = get_transformation_matrix(centroid, yaw)
        perturb_pose(transformation_matrix, data)


class BaseFrameAugmentation(DataAugmentation):
    def __init__(self, perturb_prob: float, delta_t: float, delta_yaw: float) -> None:
        assert delta_t >= 0.0 and (np.pi >= delta_yaw >= 0.0)

        self.perturb_prob = perturb_prob
        self.delta_t = delta_t
        self.delta_yaw = delta_yaw

    def _get_perturbation_matrix(self) -> np.ndarray:
        r = np.random.uniform(-self.delta_yaw, self.delta_yaw, 1)
        t = np.random.uniform(-self.delta_t, self.delta_t, (1, 2))
        return get_transformation_matrix(t, r)

    def perturb(self, data: T.Dict[str, np.ndarray]) -> None:
        if np.random.rand() >= self.perturb_prob:
            return

        perturbation_se2 = self._get_perturbation_matrix()
        perturb_pose(perturbation_se2, data)



