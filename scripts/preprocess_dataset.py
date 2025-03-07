import os

# Disable CUDA/GPU visibility before importing TensorFlow, this is important to disable
# distracting debug info.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

import pathlib
import pickle
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from functools import partial
from multiprocessing.managers import DictProxy
from typing import Final

import numpy as np
import tensorflow as tf
from rich.progress import TaskID
from waymo_open_dataset.protos import scenario_pb2

from data_utils.parallel_executor import ParallelExecutor
from data_utils.proto_decoding import generate_features_from_proto

# Global settings for Tensorflow, to avoid using GPUs and to limit the number of threads
tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

AVG_NUM_RECORDS_PER_FILE: Final[int] = 400


def _parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data-dir", "-d", type=str, required=True)
    parser.add_argument("--out-dir", "-o", type=str, required=True)
    parser.add_argument("--num-workers", "-n", type=int, default=16)
    return parser.parse_args()


def _process_file(
    file_path: pathlib.Path,
    tp_progress_dict: DictProxy,
    tp_task_id: TaskID,
    out_path: pathlib.Path,
) -> dict[str, list[int]]:
    """Process a single tfrecord file and save the processed features to disk.

    Args:
        file_path: The path to the tfrecord file to process.
        tp_progress_dict: A dictionary to store the progress of the task.
        tp_task_id: The id of the task.
        out_path: The path to save the processed features to.

    Returns:
        A dictionary mapping from scenario id to valid actor indices.
    """

    file_path_str = str(file_path)

    dataset = tf.data.TFRecordDataset(file_path_str, compression_type="")

    mapping_scenario_id_to_valid_indices: defaultdict[str, list[int]] = defaultdict(
        list
    )

    for n, payload in enumerate(dataset.as_numpy_iterator()):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(payload)

        scenario_id = scenario.scenario_id

        agent_features = generate_features_from_proto(scenario)

        tracks_to_predict_mask = agent_features["tracks_to_predict"]
        for idx, is_track_to_predict in enumerate(tracks_to_predict_mask):
            if not is_track_to_predict:
                continue
            mapping_scenario_id_to_valid_indices[scenario_id].append(idx)

        # # Save the processed features for the current scenario
        with open(out_path / f"scenario_{scenario_id}.pkl", "wb") as f:
            pickle.dump(agent_features, f)

        tp_progress_dict[tp_task_id] = {
            "progress": n + 1,
            "total": AVG_NUM_RECORDS_PER_FILE,
            "done": False,
        }

    tp_progress_dict[tp_task_id] = {
        "progress": n + 1,
        "total": AVG_NUM_RECORDS_PER_FILE,
        "done": True,
    }
    return dict(mapping_scenario_id_to_valid_indices)


def write_dataset_metadata(
    scenario_uuid_to_actors: dict[str, list[int]], output_path: pathlib.Path
) -> None:
    """
    Write the dataset metadata to disk. This includes the mapping from scenario id to valid
    actor indices, the mapping from scenario id to scenario index, the coupled indices, and
    the scenario to actor mapping.

    An example of a metadata file with a single scenario:
        scenario_uuid_to_actors = {
            'abcde': [0, 1, 2],
        },
        scenario_idx_to_uuid = {
            0: 'abcde',
        },
        scenario_uuid_to_idx = {
            'abcde': 0,
        },
        coupled_indices = [
            (0, 0),
            (0, 1),
            (0, 2),
        ],
    """
    print(
        f"Total Number of scenarios: {len(scenario_uuid_to_actors)}, writing metadata to disk ..."
    )

    scenario_idx_to_uuid = {
        idx: scenario_uuid
        for idx, scenario_uuid in enumerate(scenario_uuid_to_actors.keys())
    }

    scenario_uuid_to_idx = {
        scenario_uuid: idx for idx, scenario_uuid in scenario_idx_to_uuid.items()
    }

    coupled_indices = []
    for scenario_uuid, actors in scenario_uuid_to_actors.items():
        for actor_idx in actors:
            coupled_indices.append((scenario_uuid_to_idx[scenario_uuid], actor_idx))

    metadata = {
        "scenario_idx_to_uuid": scenario_idx_to_uuid,
        "scenario_uuid_to_idx": scenario_uuid_to_idx,
        "coupled_indices": np.array(coupled_indices),
        "scenario_uuid_to_actors": scenario_uuid_to_actors,
    }

    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("metadata written to disk")


def main(data_dir: str, out_dir: str, n_workers: int) -> None:
    datapath = pathlib.Path(data_dir)
    outpath = pathlib.Path(out_dir)
    outpath.mkdir(parents=True, exist_ok=True)

    assert datapath.is_dir()
    file_paths = list(datapath.absolute().rglob("*tfrecord*"))
    file_paths_strs = [str(file) for file in file_paths if file.is_file()]

    process_files_task = partial(_process_file, out_path=outpath)

    # Keep track of the mapping from scenario id to valid actor indices
    scenario_uuid_to_actors: dict[str, list[int]] = {}

    executor = ParallelExecutor(n_workers=n_workers)
    for mapping_per_file in executor.execute(process_files_task, file_paths_strs):
        scenario_uuid_to_actors.update(mapping_per_file)

    write_dataset_metadata(scenario_uuid_to_actors, outpath)


if __name__ == "__main__":
    args = _parse_arguments()
    main(data_dir=args.data_dir, out_dir=args.out_dir, n_workers=args.num_workers)
