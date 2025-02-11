import typing as T

import h5py
import numpy as np
from torch.utils.data import Dataset

from common_utils.tensor_utils import force_pad_batch_size
from data_utils.data_augmentation import (
    AnchorFrameAugmentation,
    BaseFrameAugmentation,
    ComposedAugmentation,
)


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


# def _generate_agent_features(
#     gt_states: np.ndarray, gt_states_avails: np.ndarray
# ) -> T.Dict[str, np.ndarray]:
#     xy = gt_states[..., :2]
#     length_and_width = gt_states[..., 2:4]
#     yaws = gt_states[..., 4, None]
#     velocities = gt_states[..., 5:7]

#     # Get the sign of the speed (is car going forwards?)
#     # speed = np.linalg.norm(velocities, axis=-1, keepdims=True)
#     # speed_sign = np.sign(velocities[..., 0, None] * np.cos(yaws) + velocities[..., 1, None] * np.sin(yaws))

#     # features = np.concatenate((xy, np.cos(yaws), np.sin(yaws), speed_sign * speed, length_and_width), axis=-1)
#     features = np.concatenate(
#         (xy, np.cos(yaws), np.sin(yaws), velocities, length_and_width), axis=-1
#     )
#     # features = np.concatenate((xy, length_and_width, yaws, velocities), axis=-1)
#     return features * gt_states_avails[..., None]


def _generate_agent_features(
    gt_states: np.ndarray, gt_states_avails: np.ndarray, delta_t: float = 0.2
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Concatenate history and future data from the batch and compute a state of the form
             {x, y, cos, sin, signed_speed}
    Returns:
     (features, availabilities) [Tensor, Tensor]
    """

    availabilities = gt_states_avails.copy()

    # Get rid of the last element of every sequence because the velocity will be corrupted
    left_shit_avails = np.concatenate(
        (availabilities[..., 1:], np.zeros_like(availabilities[..., 0, None]).astype(np.bool_)),
        axis=-1,
    )
    availabilities = np.logical_and(availabilities, left_shit_avails)

    positions = gt_states[..., :2]
    length_width = gt_states[..., 2:4]
    yaws = gt_states[..., 4, None]

    # To compute velocities do integration with the previous timestep
    # s(t+1) = s(t) + v(t) * dt
    # v(t) = (s(t+1) - s(t)) / dt
    velocities = (positions[:, 1:] - positions[:, :-1]) / delta_t
    # Append zero to last velocities
    velocities = np.concatenate((velocities, np.zeros_like(velocities[:, 0, None])), axis=-2)

    # Get the sign of the speed (is car going forwards?)
    speed = np.linalg.norm(velocities, axis=-1, keepdims=True)
    speed_sign = np.sign(
        velocities[..., 0, None] * np.cos(yaws) + velocities[..., 1, None] * np.sin(yaws)
    )

    # Get full state representation
    features = np.concatenate((positions, np.cos(yaws), np.sin(yaws), speed_sign * speed, length_width), axis=-1)
    features = features * availabilities[..., None]

    return features.astype(np.float32), availabilities.astype(np.bool_)


class ProcessedDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        data_perturb_cfg: T.Optional[T.Any] = None,
        train_with_tracks_to_predict: bool = False,
    ):
        """Do not open the h5 file here"""
        self.file: T.Optional[h5py.File] = None
        self.filepath = filepath
        self.perturb_cfg = data_perturb_cfg
        self.train_with_tracks_to_predict = train_with_tracks_to_predict

        self.augmentation = (
            ComposedAugmentation(
                [
                    AnchorFrameAugmentation(
                        perturb_prob=self.perturb_cfg.anchor_frame.perturb_prob
                    ),
                    BaseFrameAugmentation(
                        perturb_prob=self.perturb_cfg.base_frame.perturb_prob,
                        delta_t=self.perturb_cfg.base_frame.delta_t,
                        delta_yaw=self.perturb_cfg.base_frame.delta_yaw,
                    ),
                ]
            )
            if self.perturb_cfg is not None
            else None
        )

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

        # Reverse the roadgraph features in case we want to use an RNN, this simulates left padding
        # instead of right padding, TODO I still do not know if this makes a difference
        batch["roadgraph_features"] = batch["roadgraph_features"][:, ::-1].copy()
        batch["roadgraph_features_mask"] = batch["roadgraph_features_mask"][:, ::-1].copy()

        if self.augmentation is not None:
            self.augmentation.perturb(batch)

        if self.train_with_tracks_to_predict:
            batch = mask_only_target_agents_and_sdc(batch)

        # Generate the agent sequence features, these are different than gt_states and we can change
        # them to do some feature engineering
        batch["gt_features"], batch["gt_states_avails"] = _generate_agent_features(
            batch["gt_states"], batch["gt_states_avails"]
        )

        return batch

    def __del__(self):
        if self.file is not None:
            self.file.close()
