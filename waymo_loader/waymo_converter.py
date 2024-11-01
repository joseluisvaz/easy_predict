import pathlib
from argparse import ArgumentParser, Namespace
import typing as T

import h5py
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from feature_description import get_feature_description

VALIDATION_LENGTH = 44102
BATCH_SIZE = 200  # Adjust the batch size as needed

def _parse_arguments() -> Namespace:
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the folder with the tf records.")
    parser.add_argument("--out", type=str, required=True, help="Output h5 file")
    return parser.parse_args()

def _generate_records_from_files(files: T.List[str]) -> T.Generator[T.Dict[str, np.ndarray], None, None]:
    """Generates the records from the files."""
    dataset = tf.data.TFRecordDataset(files, compression_type="")
    for payload in dataset.as_numpy_iterator():
        sample = tf.io.parse_single_example(payload, get_feature_description())
        yield {k: v.numpy() for k, v in sample.items()}

def _create_datasets(file: h5py.File, new_data: T.Dict[str, np.ndarray]) -> None:
    """Create a dataset in the h5 file for every field in the new data sample."""
    for feature_name, tensor in new_data.items():
        tensor = np.expand_dims(tensor, axis=0)
        maxshape = (None, *tensor.shape[1:])
        file.create_dataset(feature_name, data=tensor, chunks=True, maxshape=maxshape)

def _append_to_datasets(file: h5py.File, batch_data: T.List[T.Dict[str, np.ndarray]]) -> None:
    """Append a batch of new data samples to the h5 file."""
    for feature_name in batch_data[0].keys():
        tensors = np.concatenate([np.expand_dims(data[feature_name], axis=0) for data in batch_data], axis=0)
        number_of_elements = file[feature_name].shape[0]
        file[feature_name].resize(number_of_elements + tensors.shape[0], axis=0)
        file[feature_name][-tensors.shape[0]:] = tensors

def _get_file_list(data_dir: str) -> T.List[str]:
    """Get file list from a data path."""
    assert pathlib.Path(data_dir).is_dir()
    return [str(file) for file in pathlib.Path(data_dir).absolute().rglob("*") if file.is_file()]

def _process_files(file_paths: T.List[str], queue: mp.Queue, progress_bar: tqdm) -> None:
    """Process files and put the results in a queue."""
    for record in _generate_records_from_files(file_paths):
        queue.put(record)
        
        progress_bar.update(1) # Update the progress bar in the secondary process 

    queue.put(None)  # Signal that processing is done

def _convert_to_h5(data_dir: str, out_path: str) -> None:
    """Convert the Waymo dataset into an h5 file that we can use for training."""
    file_paths = _get_file_list(data_dir)
    queue = mp.Queue(maxsize=10 * BATCH_SIZE)
    with tqdm(total=VALIDATION_LENGTH, desc="Processing files") as progress_bar:
        process = mp.Process(target=_process_files, args=(file_paths, queue, progress_bar))
        process.start()

        with h5py.File(out_path, "w") as f, ThreadPoolExecutor(max_workers=1) as executor:
            batch_data = []
            future = None

            while True:
                item = queue.get()
                if item is None:
                    break

                batch_data.append(item)

                if len(batch_data) == 1:
                    _create_datasets(f, item)

                if len(batch_data) >= BATCH_SIZE:
                    if future:
                        future.result()  # Wait for the previous batch to be processed
                    future = executor.submit(_append_to_datasets, f, batch_data)
                    batch_data = []  # Reset the batch data

            if batch_data:
                if future:
                    future.result()  # Ensure the previous batch is processed
                _append_to_datasets(f, batch_data)

        process.join()
    

if __name__ == "__main__":
    args = _parse_arguments()
    _convert_to_h5(args.data_dir, args.out)
