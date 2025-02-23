import typing as T
from argparse import ArgumentParser, Namespace

import h5py
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.feature_description import MAX_AGENTS_TO_PREDICT
from data_utils.feature_generation import WaymoH5Dataset, collate_waymo_stack

BATCH_SIZE = 256
NUM_WORKERS = 16


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the h5 file to process"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output h5 file with the processed data"
    )
    return parser.parse_args()


def _create_h5_datasets(file: h5py.File, batch_data: T.Dict[str, np.ndarray]) -> None:
    """Create a dataset in the h5 file for every field in the new data sample."""
    for dataset_name, first_data in batch_data.items():
        maxshape = (None, *first_data.shape[1:])
        file.create_dataset(
            dataset_name,
            data=first_data,
            chunks=True,
            maxshape=maxshape,
            compression="gzip",
        )


def _append_to_h5_datasets(
    file: h5py.File, batch_data: T.Dict[str, np.ndarray]
) -> None:
    """Append a batch of new data samples to the h5 file."""
    for dataset_name, data in batch_data.items():
        number_of_elements = file[dataset_name].shape[0]
        file[dataset_name].resize(number_of_elements + data.shape[0], axis=0)
        file[dataset_name][-data.shape[0] :] = data


def main(data_dir: str, out: str):
    # Set train_with_tracks_to_predict to False to generate the data with the 128 full agent data
    dataset = WaymoH5Dataset(data_dir, train_with_tracks_to_predict=False)
    dataloader = DataLoader(
        dataset,
        batch_size=12,
        num_workers=NUM_WORKERS,
        shuffle=False,
        persistent_workers=True,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=4,
        collate_fn=collate_waymo_stack,
    )

    with h5py.File(out, "w") as file:
        dataset_created_flag = False
        for batch in tqdm(dataloader):
            if not dataset_created_flag:
                _create_h5_datasets(file, batch)
                dataset_created_flag = True
            else:
                _append_to_h5_datasets(file, batch)

        # Add a relational table in the dataset that includes the scenario_id to agent_id mapping
        num_scenarios = len(file["gt_states"])

        coupled_indices = []
        for dataset_idx in tqdm(range(num_scenarios), total=num_scenarios):
            scenario_id = file["scenario_id"][dataset_idx]
            assert scenario_id >= 0
            tracks_to_predict_mask = np.array(
                file["tracks_to_predict"][dataset_idx]
            ).astype(np.bool_)

            for agent_id in range(MAX_AGENTS_TO_PREDICT):
                if not tracks_to_predict_mask[agent_id]:
                    continue
                coupled_index = np.array([scenario_id, agent_id]).astype(np.int64)
                coupled_indices.append(coupled_index)

        file.create_dataset(
            "coupled_indices",
            data=np.stack(coupled_indices, axis=0),
            compression="gzip",
            chunks=True,
        )


if __name__ == "__main__":
    args = _parse_arguments()
    main(args.data_dir, args.out)
