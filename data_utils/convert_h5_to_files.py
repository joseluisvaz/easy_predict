import pathlib
import typing as T
from argparse import ArgumentParser, Namespace

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_utils.feature_description import MAX_AGENTS_TO_PREDICT 
from data_utils.processed_dataset import _get_scenario_from_h5_file_using_idx


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the h5 file to process"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output h5 file with the processed data"
    )
    return parser.parse_args()


class PickledDataset(Dataset):
    def __init__(
        self,
        filepath: str,
    ):
        self.filepath = filepath
        self.file: T.Optional[h5py.File] = None

        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as file:
            self.dataset_len = len(file["gt_states"])  # Example version of the dataset


    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, scenario_idx: int) -> T.Dict[str, np.ndarray]:
        if self.file is None:
            self.file = h5py.File(self.filepath, "r", libver="latest", swmr=True)

        return _get_scenario_from_h5_file_using_idx(self.file, scenario_idx)

    def __del__(self):
        if self.file is not None:
            self.file.close()


def main(data_dir: str, out: str) -> None:

    dataset = PickledDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=16,
        shuffle=False,
        persistent_workers=True,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=4,
        collate_fn=lambda x: x[0],  # No collate function needed
    )

    path = pathlib.Path(out)
    path.mkdir(parents=True, exist_ok=True)

    coupled_indices = []

    for sample in tqdm(dataloader):
        scenario_id = sample["scenario_id"].item()
        filepath = path / f"batch_{scenario_id}.pt"
        torch.save(sample, str(filepath))

        for agent_id in range(MAX_AGENTS_TO_PREDICT):
            if not sample["tracks_to_predict"][agent_id]:
                continue
            coupled_indices.append(np.array([scenario_id, agent_id]))

    coupled_indices = np.stack(coupled_indices, axis=0).astype(np.int64)
    torch.save(coupled_indices, path / "coupled_indices.pt")


if __name__ == "__main__":
    args = _parse_arguments()
    main(args.data_dir, args.out)
