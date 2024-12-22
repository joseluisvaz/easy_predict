import typing as T

import h5py
import numpy as np
from torch.utils.data import Dataset

class ProcessedDataset(Dataset):
    def __init__(self, filepath: str):
        """Do not open the h5 file here"""
        self.file: T.Optional[h5py.File] = None
        self.filepath = filepath

        # Open and close dataset just to extract the length
        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as file:
            self.dataset_len = len(file["gt_states"])  # Example version of the dataset

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> T.Dict[str, np.ndarray]:
        # NOTE: We open the dataset here so that each worker has its own file handle
        if self.file is None:
            self.file = h5py.File(self.filepath, "r", libver="latest", swmr=True)

        return {
            "gt_states": np.array(self.file["gt_states"][idx]).astype(np.float32), # [N_AGENTS, TIME, FEATS]
            "gt_states_avails": np.array(self.file["gt_states_avails"][idx]).astype(np.bool_), # [N_AGENTS, TIME]
            "actor_type": np.array(self.file["actor_type"][idx]).astype(np.int64), # [N_AGENTS,]
            "is_sdc": np.array(self.file["is_sdc"][idx]).astype(np.bool_), # [N_AGENTS,]
            "tracks_to_predict": np.array(self.file["tracks_to_predict"][idx]).astype(np.bool_), # [N_AGENTS,]
            "roadgraph_features": np.array(self.file["roadgraph_features"][idx]).astype(np.float32), # [N_POLYLINE, N_POINTS, FEATS]
            "roadgraph_features_mask": np.array(self.file["roadgraph_features_mask"][idx]).astype(np.bool_), # [N_POLYLINE, N_POINTS]
            "roadgraph_features_types": np.array(self.file["roadgraph_features_types"][idx]).astype(np.int64), # [N_POLYLINE,]
            "roadgraph_features_ids": np.array(self.file["roadgraph_features_ids"][idx]).astype(np.int16), # [N_POLYLINE,]
        }

    def __del__(self):
        if self.file is not None:
            self.file.close()
