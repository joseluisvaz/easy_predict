from multiprocessing.managers import DictProxy

import pytest
from rich.progress import TaskID

from data_utils.parallel_executor import ParallelExecutor


def _square(x: int, tp_progress_dict: DictProxy, tp_task_id: TaskID) -> int:
    tp_progress_dict[tp_task_id] = {"progress": 2, "total": 5}
    return x * x


def _square_with_error(x: int, tp_progress_dict: DictProxy, tp_task_id: TaskID) -> int:
    tp_progress_dict[tp_task_id] = {"progress": 2, "total": 5}
    raise ValueError("Error processing value")


@pytest.mark.parametrize(
    "n_workers",
    [
        1,
        2,
    ],
)
def test_successful_parallel_execution(n_workers: int) -> None:
    executor = ParallelExecutor(n_workers=n_workers)
    inputs = [4, 5]
    results = list(executor.execute(_square, inputs))
    assert results == [16, 25]


def test_unsuccessful_parallel_execution() -> None:
    executor = ParallelExecutor(n_workers=2)
    inputs = [4, 5]
    with pytest.raises(ValueError, match="Error processing value"):
        for _ in executor.execute(_square_with_error, inputs):
            pass


def test_empty_input_list() -> None:
    executor = ParallelExecutor(n_workers=2)
    results = list(executor.execute(_square, []))
    assert results == []
