import typing as T

import h5py
import numpy as np
from torch.utils.data import Dataset

from common_utils.geometry import (
    get_so2_from_se2,
    get_transformation_matrix,
    get_yaw_from_se2,
    transform_points,
)


def perturb_pose(batch: T.Dict[str, np.ndarray], delta_yaw: float = 0.1, delta_t: float = 1.0):
    """Perturb the pose of the agents in the batch"""
    r = np.random.uniform(-delta_yaw, delta_yaw, 1)
    t = np.random.uniform(-delta_t, delta_t, (1, 2))
    perturbation_se2 = get_transformation_matrix(t, r)
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


class ProcessedDataset(Dataset):
    def __init__(self, filepath: str, data_perturb_cfg: T.Optional[T.Any] = None):
        """Do not open the h5 file here"""
        self.file: T.Optional[h5py.File] = None
        self.filepath = filepath
        self.perturb_cfg = data_perturb_cfg

        # Open and close dataset just to extract the length
        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as file:
            self.dataset_len = len(file["gt_states"])  # Example version of the dataset

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> T.Dict[str, np.ndarray]:
        # NOTE: We open the dataset here so that each worker has its own file handle
        if self.file is None:
            self.file = h5py.File(self.filepath, "r", libver="latest", swmr=True)

        batch = {
            "gt_states": np.array(self.file["gt_states"][idx]).astype(
                np.float32
            ),  # [N_AGENTS, TIME, FEATS]
            "gt_states_avails": np.array(self.file["gt_states_avails"][idx]).astype(
                np.bool_
            ),  # [N_AGENTS, TIME,]
            "actor_type": np.array(self.file["actor_type"][idx]).astype(np.int64),  # [N_AGENTS,]
            "is_sdc": np.array(self.file["is_sdc"][idx]).astype(np.bool_),  # [N_AGENTS,]
            "tracks_to_predict": np.array(self.file["tracks_to_predict"][idx]).astype(
                np.bool_
            ),  # [N_AGENTS,]
            "roadgraph_features": np.array(self.file["roadgraph_features"][idx]).astype(
                np.float32
            ),  # [N_POLYLINE, N_POINTS, FEATS]
            "roadgraph_features_mask": np.array(self.file["roadgraph_features_mask"][idx]).astype(
                np.bool_
            ),  # [N_POLYLINE, N_POINTS]
            "roadgraph_features_types": np.array(self.file["roadgraph_features_types"][idx]).astype(
                np.int64
            ),  # [N_POLYLINE,]
            "roadgraph_features_ids": np.array(self.file["roadgraph_features_ids"][idx]).astype(
                np.int16
            ),  # [N_POLYLINE,]
        }

        batch["roadgraph_features"] = batch["roadgraph_features"][:, ::-1].copy()
        batch["roadgraph_features_mask"] = batch["roadgraph_features_mask"][:, ::-1].copy()

        if (self.perturb_cfg is not None) and (np.random.rand() < self.perturb_cfg.perturb_prob):
            batch = perturb_pose(batch, self.perturb_cfg.delta_yaw, self.perturb_cfg.delta_t)

        return batch

    def __del__(self):
        if self.file is not None:
            self.file.close()
