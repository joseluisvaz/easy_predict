import io
import itertools
import typing as T
from dataclasses import dataclass

import lightning as L
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchinfo import summary

from common_utils.visualization import plot_scene
from data_utils.feature_description import NUM_HISTORY_FRAMES
from data_utils.feature_generation import collate_waymo
from data_utils.processed_dataset import ProcessedDataset
from models.inference import run_model_forward_pass


class OnTrainCallback(L.Callback):
    def __init__(self, datadir: str):
        super().__init__()

        self._n_samples = 5
        dataset = ProcessedDataset(datadir, train_with_tracks_to_predict=True)
        self._dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_waymo,
        )

    def on_train_start(self, trainer, pl_module):
        batch = next(iter(self._dataloader))
        batch = {k: v.cuda() for k, v in batch.items()}

        history_states = batch["gt_features"][:, :, : NUM_HISTORY_FRAMES + 1]
        history_avails = batch["gt_states_avails"][:, :, : NUM_HISTORY_FRAMES + 1]
        summary(
            pl_module.model,
            input_data=(
                history_states,
                history_avails,
                batch["actor_type"],
                batch["roadgraph_features"],
                batch["roadgraph_features_mask"],
                batch["roadgraph_features_types"],
            ),
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        for idx, single_sample_batch in enumerate(
            itertools.islice(self._dataloader, 10, 10 + self._n_samples)
        ):
            # Just extract a single sample from the batch and keep the batch dimension
            single_sample_batch = {k: v.cuda() for k, v in single_sample_batch.items()}

            predicted_positions = run_model_forward_pass(pl_module.model, single_sample_batch)
            self.log_plot_to_clearml(
                pl_module,
                pl_module.task.get_logger() if pl_module.task is not None else None,
                single_sample_batch,
                predicted_positions,
                trainer.current_epoch,
                idx,
            )

        train_metric_values = pl_module.metrics.result()
        for i, m in enumerate(["min_ade", "min_fde", "miss_rate", "overlap_rate", "map"]):
            for j, n in enumerate(pl_module.metrics.metric_names):
                self.log(f"metrics/{m}/{n}", float(train_metric_values[i, j]))
        pl_module.metrics.reset_state()

    def log_plot_to_clearml(
        self,
        pl_module: L.LightningModule,
        logger: T.Optional[T.Any],
        single_sample: T.Dict[str, torch.Tensor],
        predicted_positions: torch.Tensor,
        current_epoch: int,
        scene_idx: int,
    ):
        """Creates the image of a single scenario and logs it to ClearML.
        Assumption: Each component of the scenario has the batch dimension of size 1
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
                title="sample", series=f"scene{scene_idx}", iteration=current_epoch, image=image
            )
        else:
            image.save(f"data/visualizations/scene_{scene_idx}.png")
