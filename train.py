from argparse import ArgumentParser, Namespace
import warnings
from typing import Dict, Optional, Any

# Ignore the warning about nested tensors to not spam the terminal
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.",
)

import torch

torch.autograd.set_detect_anomaly(True)

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from clearml import Task
import lightning as L
from lightning.pytorch.tuner import Tuner
from metrics import MotionMetrics, _default_metrics_config
from metrics_callback import OnTrainCallback
from torch import Tensor

from waymo_loader.dataloaders import WaymoH5Dataset, collate_waymo
from models.prediction import PredictionModel
from waymo_loader.feature_description import (
    NUM_HISTORY_FRAMES,
    NUM_FUTURE_FRAMES,
)


plt.style.use("dark_background")
torch.set_float32_matmul_precision("medium")


def compute_loss(predicted_positions, target_positions, target_availabilities):
    errors = torch.sum((target_positions - predicted_positions) ** 2, dim=-1)
    return torch.mean(errors * target_availabilities)


task = Task.init(project_name="TrajectoryPrediction", task_name="SimpleAgentPrediction MCG")


class LightningModule(L.LightningModule):
    def __init__(
        self, data_dir: str, fast_dev_run: bool, hyperparameters: Dict[str, Any] 
    ):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.data_dir = data_dir
        self.fast_dev_run = fast_dev_run
        self.learning_rate = hyperparameters["learning_rate"] 
        self.n_timesteps = 80
        self.batch_size = hyperparameters["batch_size"]
        self.model = PredictionModel(
            input_features=12, hidden_size=hyperparameters["hidden_size"], n_timesteps=self.n_timesteps
        )

        self.metrics_config = _default_metrics_config()
        self.metrics = MotionMetrics(self.metrics_config)

    def _update_metrics(self, batch: Dict[str, Tensor], predicted_positions: Tensor):
        batch_size, num_agents, _, _ = predicted_positions.shape
        # [batch_size, num_agents, steps, 2] -> # [batch_size, 1, 1, num_agents, steps, 2].
        # The added dimensions are top_k = 1, num_agents_per_joint_prediction = 1.
        predicted_positions = predicted_positions[:, None, None]
        # Fake the score since this model does not generate any score per predicted
        # trajectory. Get the first shapes [batch_size, num_preds, top_k] -> [batch_size, 1, 1].
        pred_score = torch.ones((batch_size, 1, 1))
        # [batch_size, num_pred, num_agents].
        pred_gt_indices = torch.arange(num_agents, dtype=torch.int64)
        pred_gt_indices = pred_gt_indices[None, None, :].expand(batch_size, 1, num_agents)
        # For the tracks to predict use the current timestamps
        pred_gt_indices_mask = batch["gt_states_avails"][:, :, NUM_HISTORY_FRAMES]
        pred_gt_indices_mask = pred_gt_indices_mask.unsqueeze(1)

        self.metrics.update_state(
            prediction_trajectory=predicted_positions,
            prediction_score=pred_score,
            ground_truth_trajectory=batch["gt_states"],
            ground_truth_is_valid=batch["gt_states_avails"],
            prediction_ground_truth_indices=pred_gt_indices,
            prediction_ground_truth_indices_mask=pred_gt_indices_mask,
            object_type=batch["actor_type"][..., 0],
        )

    def _inference_and_loss(self, batch):
        history_states = batch["gt_states"][:, :, : NUM_HISTORY_FRAMES + 1, :]
        history_avails = batch["gt_states_avails"][:, :, : NUM_HISTORY_FRAMES + 1]

        predicted_positions = self.model(
            history_states,
            history_avails,
            batch["actor_type"],
            batch["roadgraph_features"],
            batch["roadgraph_features_mask"],
        )

        # Crop the future positions to match the number of timesteps
        future_positions = batch["gt_states"][:, :, -NUM_FUTURE_FRAMES:, :2]
        future_availabilities = batch["gt_states_avails"][:, :, -NUM_FUTURE_FRAMES:]
        loss = compute_loss(predicted_positions, future_positions, future_availabilities)
        return predicted_positions, loss

    def training_step(self, batch, batch_idx):

        predicted_positions, loss = self._inference_and_loss(batch)

        self._update_metrics(batch, predicted_positions)

        # Compute the percentarge of elements in the map that are available, this ia metric that tells us
        # how empty the tensors are
        map_avails = torch.nested.to_padded_tensor(
            batch["roadgraph_features_mask"], padding=False
        ).bool()[
            ..., 0
        ]  # batch, polyline, points

        percentage = map_avails.sum().float() / map_avails.numel()
        self.log("map_availability_percentage", percentage)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss = self._inference_and_loss(batch)
        self.log("loss/val", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        dataset = WaymoH5Dataset(self.data_dir)
        return DataLoader(
            dataset,
            batch_size=self.batch_size if not self.fast_dev_run else 8,
            num_workers=8 if not self.fast_dev_run else 1,
            shuffle=True,
            persistent_workers=True if not self.fast_dev_run else False,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=8 if not self.fast_dev_run else None,
            collate_fn=collate_waymo,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.dataset,
    #         batch_size=self.batch_size if not self.fast_dev_run else 8,
    #         num_workers=8 if not self.fast_dev_run else 1,
    #         shuffle=False,
    #         persistent_workers=True if not self.fast_dev_run else False,
    #         pin_memory=False,
    #         drop_last=True,
    #         prefetch_factor=8 if not self.fast_dev_run else None,
    #         collate_fn=collate_waymo,
    #     )


def main(data_dir: str, fast_dev_run: bool, use_gpu: bool, ckpt_path: Optional[str]):
    dataset = WaymoH5Dataset(data_dir)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    hyperparameters = {"batch_size": 128, "learning_rate": 0.0002, "hidden_size": 128}
    task.connect(hyperparameters)

    # Load if a checkpoint is provided
    module = (
        LightningModule(data_dir, fast_dev_run, hyperparameters)
        if not ckpt_path
        else LightningModule.load_from_checkpoint(
            ckpt_path,
            data_dir=data_dir,
            fast_dev_run=fast_dev_run,
            hyperparameters=hyperparameters,
            map_location="cpu",
        )
    )

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        fast_dev_run=fast_dev_run,
        # precision="16-mixed",
        callbacks=OnTrainCallback(data_dir),
    )

    LR_FIND = False
    if LR_FIND and not fast_dev_run:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(module)

        fig = lr_finder.plot(suggest=True)
        fig.savefig("learning_rate.png")
        new_lr = lr_finder.suggestion()
        print("LEARNING RATE SUGGESTION: ", new_lr)

    trainer.fit(model=module)


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the folder with the tf records."
    )
    parser.add_argument("--fast-dev-run", action="store_true", help="Fast dev run")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--ckpt", required=False, type=str, help="Checkpoint file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    main(args.data_dir, args.fast_dev_run, args.gpu, args.ckpt)
