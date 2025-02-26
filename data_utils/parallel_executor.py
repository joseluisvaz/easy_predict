import multiprocessing as mp
import typing as T
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing.managers import DictProxy
from typing import Any, Callable

import rich.progress as rp


class ParallelExecutor:
    def __init__(self, n_workers: int):
        """Initialize the thread pool executor with progress.

        Args:
            n_workers: The number of workers to use.
        """
        self.n_workers = n_workers

    def _submit_tasks(
        self,
        fn: Callable,
        iterable: T.Iterable,
        progress_manager: rp.Progress,
        tp_progress_dict: DictProxy,
        executor: ProcessPoolExecutor,
    ) -> T.List[Future]:
        """Submit tasks to the executor and return a list of futures.

        Args:
            fn: The function to execute on the iterable items..
            iterable: The iterable to process and submit and spawn tasks.
            tp_progress_dict: A dictionary to store the progress of each task.
            executor: The executor to use.

        Returns:
            A list of futures.
        """
        futures = []
        for n, item in enumerate(iterable):
            task_id = progress_manager.add_task(f"task {n}", visible=False)
            futures.append(
                executor.submit(
                    fn, item, tp_progress_dict=tp_progress_dict, tp_task_id=task_id
                )
            )
        return futures

    def _monitor_progress(
        self,
        futures: T.List[Future],
        progress_manager: rp.Progress,
        main_process_task_id: rp.TaskID,
        tp_progress_dict: DictProxy,
    ) -> None:
        """Monitor the progress of the tasks and update the overall progress bar.

        Args:
            futures: A list of futures waiting to be completed.
            progress_manager: The progress manager to use.
            overall_progress_task: The task to update for the main progress bar.
            tp_progress_dict: A dictionary with the stored progress of each task.
        """
        while (n_finished := sum([future.done() for future in futures])) < len(futures):
            # Update the overall progress bar
            progress_manager.update(
                main_process_task_id,
                completed=n_finished,
                total=len(futures),
            )

            for task_id, update_data in tp_progress_dict.items():
                latest = update_data["progress"]
                total = update_data["total"]
                done = update_data["done"]
                progress_manager.update(
                    task_id,
                    completed=latest,
                    total=total,
                    visible=not done,
                )

    def execute(
        self, fn: Callable, iterable: T.Iterable
    ) -> T.Generator[Any, None, None]:
        """
        Map a function over an iterable, with progress reporting.

        Args:
            fn: The function to execute on the iterable items.
            iterable: The iterable to process and submit and spawn tasks.

        Returns:
            A list of results from the function.
        """
        with (
            rp.Progress(
                "[progress.description]{task.description}",
                rp.SpinnerColumn(),
                rp.BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.0f}%",
                rp.TimeRemainingColumn(),
                rp.TimeElapsedColumn(),
                refresh_per_second=10,
                expand=True,
            ) as progress_manager,
            mp.Manager() as multiprocessing_manager,
            ProcessPoolExecutor(max_workers=self.n_workers) as executor,
        ):
            main_process_task_id = progress_manager.add_task(
                "[green]All jobs progress:"
            )
            tp_progress_dict = multiprocessing_manager.dict()

            # Submit all tasks required for execution
            futures = self._submit_tasks(
                fn, iterable, progress_manager, tp_progress_dict, executor
            )

            # Monitor the progress of the tasks and update the multiple progress bars
            self._monitor_progress(
                futures,
                progress_manager,
                main_process_task_id,
                tp_progress_dict,
            )

            # Yield the results of the tasks as they are completed
            for future in futures:
                yield future.result()
