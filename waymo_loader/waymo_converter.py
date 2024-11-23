import pathlib
from argparse import ArgumentParser, Namespace
import typing as T

import h5py
import zarr
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from feature_description import get_feature_description, STATE_FEATURES

VALIDATION_LENGTH = 44102
BATCH_SIZE = 200  # Adjust the batch size as needed
MAX_AGENTS = 128

FEATURE_DESCRIPTION: T.Dict = dict(STATE_FEATURES)


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the folder with the tf records."
    )
    parser.add_argument("--out", type=str, required=True, help="Output h5 file")
    return parser.parse_args()


def _generate_records_from_files(
    files: T.List[str], max_agents: int = MAX_AGENTS
) -> T.Generator[T.Dict[str, np.ndarray], None, None]:
    """Generates the records from the files."""
    dataset = tf.data.TFRecordDataset(files, compression_type="")
    for payload in dataset.as_numpy_iterator():
        decoded_example = tf.io.parse_single_example(payload, get_feature_description())
        processed_example = {}

        # Convert tensors to numpy arrays
        for key, value in decoded_example.items():
            decoded_example[key] = value.numpy()

        # Apply the tracks_to_predict mask
        # track_mask = decoded_example["state/tracks_to_predict"] > 0
        # sdc_mask = decoded_example["state/is_sdc"] > 0
        # mask = track_mask & sdc_mask

        for key, value in decoded_example.items():
            if key not in STATE_FEATURES:
                continue
            # TODO: Implement parsing of map features
            # cropped_value = value[mask]
            # # Pad the cropped value to max_agents
            # if cropped_value.shape[0] < max_agents:
            #     padding = ((0, max_agents - cropped_value.shape[0]),) + ((0, 0),) * (
            #         cropped_value.ndim - 1
            #     )
            #     padded_value = np.pad(cropped_value, padding, mode="constant", constant_values=0)
            # else:
            #     padded_value = cropped_value[:max_agents]

            # For the features with a single dimension, add an extra for concatenation
            if len(value.shape) == 1:
                value = np.expand_dims(value, axis=1)

            processed_example[key] = value

        # Iterate ver the ordered features and create a vector
        feature_sequence = []
        for key in STATE_FEATURES:
            feature_sequence.append(processed_example[key])

        merged_features = np.concatenate(feature_sequence, axis=-1)
        yield merged_features


def _process_files(file_paths: T.List[str], queue: mp.Queue, progress_bar: tqdm) -> None:
    """Process files and put the results in a queue."""
    for record in _generate_records_from_files(file_paths):
        queue.put(record)
        progress_bar.update(1)  # Update the progress bar in the secondary process
    queue.put(None)  # Signal that processing is done


def _get_file_list(data_dir: str) -> T.List[str]:
    """Get file list from a data path."""
    assert pathlib.Path(data_dir).is_dir()
    return [str(file) for file in pathlib.Path(data_dir).absolute().rglob("*") if file.is_file()]


def _create_h5_datasets(
    file: h5py.File, new_data: np.ndarray, max_agents: int = MAX_AGENTS
) -> None:
    """Create a dataset in the h5 file for every field in the new data sample."""
    tensor = np.expand_dims(new_data, axis=0)  # Add batch dimension
    maxshape = (None, max_agents, *tensor.shape[2:])
    file.create_dataset(
        "merged_features", data=tensor, chunks=True, maxshape=maxshape, compression="gzip"
    )


def _append_to_h5_datasets(file: h5py.File, batch_data: T.List[np.ndarray]) -> None:
    """Append a batch of new data samples to the h5 file."""
    tensors = [np.expand_dims(data, axis=0) for data in batch_data]  # Add batch dimension
    tensors = np.concatenate(tensors, axis=0)
    number_of_elements = file["merged_features"].shape[0]
    file["merged_features"].resize(number_of_elements + tensors.shape[0], axis=0)
    file["merged_features"][-tensors.shape[0] :] = tensors


def _convert_to_h5(data_dir: str, out_path: str) -> None:
    """Convert the Waymo dataset into an h5 file that we can use for training."""
    file_paths = _get_file_list(data_dir)
    queue = mp.Queue(maxsize=10 * BATCH_SIZE)
    with tqdm(total=VALIDATION_LENGTH, desc="Processing files") as progress_bar:
        process = mp.Process(target=_process_files, args=(file_paths, queue, progress_bar))
        process.start()

        with h5py.File(out_path, "w") as f:
            batch_data = []
            dataset_created_flag = False

            while True:
                item = queue.get()
                if item is None:
                    break

                batch_data.append(item)

                if len(batch_data) == 1 and dataset_created_flag is False:
                    _create_h5_datasets(f, item)
                    dataset_created_flag = True

                if len(batch_data) >= BATCH_SIZE:
                    _append_to_h5_datasets(f, batch_data)
                    batch_data = []  # Reset the batch data

            if batch_data:
                _append_to_h5_datasets(f, batch_data)

        process.join()


if __name__ == "__main__":
    args = _parse_arguments()
    _convert_to_h5(args.data_dir, args.out)
