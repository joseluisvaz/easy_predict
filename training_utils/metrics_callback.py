import io
import itertools
import typing as T

import lightning as L
import torch
from matplotlib import pyplot as plt
from PIL import Image

from models.inference import run_model_forward_pass
from utils.visualization import plot_scene


class ModelInspectionCallback(L.Callback):
    def __init__(self, viz_scenario_offset: int, viz_num_scenarios: int):
        """Callback to log the visualizations and metrics to the experiment manager logger.
        Args:
            viz_scenario_offset: The scenario offset to start the visualization from.
            num_visualization_scenarios: The number of scenarios to visualize.
        """
        super().__init__()

        self.viz_scenario_offset = viz_scenario_offset
        self.viz_num_scenarios = viz_num_scenarios

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Logs the visualizations and metrics to the experiment manager logger.

        Args:
            trainer: The Trainer instance.
            pl_module: The LightningModule instance.
        """
        self._log_visualizations(trainer, pl_module)
        self._log_metrics(pl_module)

    def _log_metrics(self, pl_module: L.LightningModule) -> None:
        """Logs the metrics to the ClearML logger.

        Assumption: The metrics are already computed in the metrics member of the pl_module.

        Args:
            pl_module: The LightningModule instance.
        """
        train_metric_values = pl_module.metrics.result()
        for i, m in enumerate(
            ["min_ade", "min_fde", "miss_rate", "overlap_rate", "map"]
        ):
            for j, n in enumerate(pl_module.metrics.metric_names):
                self.log(f"metrics/{m}/{n}", float(train_metric_values[i, j]))
        pl_module.metrics.reset_state()

    def _log_visualizations(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Logs the visualizations to the ClearML logger.

        Assumption: The visualizations are already computed in the visualizations member of the pl_module.

        Args:
            trainer: The Trainer instance.
            pl_module: The LightningModule instance.
        """
        visualization_dataloader = trainer.datamodule.visualization_dataloader()

        for idx, single_sample_batch in enumerate(
            itertools.islice(
                visualization_dataloader,
                self.viz_scenario_offset,
                self.viz_scenario_offset + self.viz_num_scenarios,
            )
        ):
            # Just extract a single sample from the batch and keep the batch dimension
            single_sample_batch = {k: v.cuda() for k, v in single_sample_batch.items()}

            predicted_positions = run_model_forward_pass(
                pl_module.model, single_sample_batch
            )
            self._log_scenario_plot_to_experiment_manager(
                pl_module.task.get_logger() if pl_module.task is not None else None,
                single_sample_batch,
                predicted_positions,
                trainer.current_epoch,
                idx,
            )

    def _log_scenario_plot_to_experiment_manager(
        self,
        logger: T.Optional[T.Any],
        single_sample: T.Dict[str, torch.Tensor],
        predicted_positions: torch.Tensor,
        current_epoch: int,
        scene_idx: int,
    ) -> None:
        """Creates the image of a single scenario and logs it to ClearML.

        Args:
            logger: The experiment manager logger.
            single_sample: A single sample from the validation set.
            predicted_positions: The predicted positions of the sample.
            current_epoch: The current epoch.
            scene_idx: The index of the scenario.
        """

        plot_scene(single_sample, predicted_positions, zoom_out=True)

        # Save the plot to an in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)

        # Convert the buffer to a PIL Image
        image = Image.open(buf)
        if image.mode == "RGBA":
            image = image.convert("RGB")

        if logger is not None:
            logger.report_image(
                title="sample",
                series=f"scene{scene_idx}",
                iteration=current_epoch,
                image=image,
            )
        else:
            image.save(f"data/visualizations/scene_{scene_idx}.png")
