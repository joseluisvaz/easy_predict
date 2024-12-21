from argparse import ArgumentParser, Namespace
import warnings
import typing as T
from pytorch_lightning.profilers import SimpleProfiler

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
from omegaconf import OmegaConf, DictConfig
import lightning as L
from lightning.pytorch.tuner import Tuner
from metrics import MotionMetrics, _default_metrics_config
from metrics_callback import OnTrainCallback
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.callbacks import ModelCheckpoint
from waymo_loader.dataloaders import WaymoH5Dataset, collate_waymo
from models.prediction import PredictionModel
from waymo_loader.feature_description import (
    NUM_HISTORY_FRAMES,
    NUM_FUTURE_FRAMES,
    SUBSAMPLE_SEQUENCE,
)

plt.style.use("dark_background")
torch.set_float32_matmul_precision("medium")

LR_FIND = False


def compute_loss(
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor,
    target_availabilities: torch.Tensor,
    tracks_to_predict_mask: torch.Tensor,
    is_sdc: torch.Tensor,
    loss_tracks_to_predict_mask: bool,
    loss_use_ego_vehicle_mask: bool,
):

    total_mask = target_availabilities

    # Use masks to hide agents out
    if loss_use_ego_vehicle_mask and loss_tracks_to_predict_mask:
        total_mask *= tracks_to_predict_mask.unsqueeze(-1) | is_sdc.unsqueeze(-1)
    elif loss_use_ego_vehicle_mask:
        total_mask *= is_sdc.unsqueeze(-1)
    elif loss_tracks_to_predict_mask:
        total_mask *= tracks_to_predict_mask.unsqueeze(-1)

    errors = torch.sum((target_positions - predicted_positions) ** 2, dim=-1)
    return torch.mean(errors * total_mask)


task = Task.init(project_name="TrajectoryPrediction", task_name="SimpleAgentPrediction MCG")

MAX_PREDICT_AGENTS = 8


class LightningModule(L.LightningModule):
    def __init__(self, data_dir: str, fast_dev_run: bool, hyperparameters: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.data_dir = data_dir
        self.fast_dev_run = fast_dev_run
        self.learning_rate = hyperparameters.learning_rate
        self.n_timesteps = NUM_FUTURE_FRAMES
        self.batch_size = hyperparameters.batch_size
        self.model = PredictionModel(
            input_features=12,
            hidden_size=hyperparameters.hidden_size,
            n_timesteps=self.n_timesteps,
        )
        self.hyperparameters = hyperparameters

        self.metrics_config = _default_metrics_config()
        self.metrics = MotionMetrics(self.metrics_config)

    def _update_metrics(self, batch: T.Dict[str, Tensor], full_predicted_positions: Tensor):

        # Chop all the tensors to match the number of predicted agents the tracks to predict mask
        # will take care of just computing the metrics for the relevant agents
        predicted_positions = full_predicted_positions[:, :MAX_PREDICT_AGENTS]
        gt_states_avails = batch["gt_states_avails"][:, :MAX_PREDICT_AGENTS]
        gt_states = batch["gt_states"][:, :MAX_PREDICT_AGENTS]
        actor_type = batch["actor_type"][:, :MAX_PREDICT_AGENTS]
        tracks_to_predict_mask = batch["tracks_to_predict"][:, :MAX_PREDICT_AGENTS]

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
        pred_gt_indices_mask = tracks_to_predict_mask
        pred_gt_indices_mask = pred_gt_indices_mask.unsqueeze(1)

        self.metrics.update_state(
            prediction_trajectory=predicted_positions,
            prediction_score=pred_score,
            ground_truth_trajectory=gt_states,
            ground_truth_is_valid=gt_states_avails,
            prediction_ground_truth_indices=pred_gt_indices,
            prediction_ground_truth_indices_mask=pred_gt_indices_mask,
            object_type=actor_type[..., 0],
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
            batch["roadgraph_features_types"],
        )

        # Crop the future positions to match the number of timesteps
        future_positions = batch["gt_states"][:, :, -NUM_FUTURE_FRAMES:, :2]
        future_availabilities = batch["gt_states_avails"][:, :, -NUM_FUTURE_FRAMES:]
        loss = compute_loss(
            predicted_positions,
            future_positions,
            future_availabilities,
            batch["tracks_to_predict"],
            batch["is_sdc"],
            self.hyperparameters.loss_tracks_to_predict_mask,
            self.hyperparameters.loss_use_ego_vehicle_mask,
        )
        return predicted_positions, loss

    def training_step(self, batch, batch_idx):

        predicted_positions, loss = self._inference_and_loss(batch)

        with torch.no_grad():
            self._update_metrics(batch, predicted_positions)

        # Compute the percentarge of elements in the map that are available, this ia metric that tells us
        # how empty the tensors are
        map_avails = torch.nested.to_padded_tensor(
            batch["roadgraph_features_mask"], padding=False
        ).bool()[
            ..., 0
        ]  # batch, polyline, points

        _, n_polyline, _ = map_avails.shape
        self.log("map_sizes/polyline", n_polyline)

        percentage = map_avails.sum().float() / map_avails.numel()
        self.log("map_availability_percentage", percentage)
        self.log("loss/train", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.hyperparameters.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer,
                              T_max=self.hyperparameters.max_epochs * len(self.train_dataloader()),
                              eta_min=self.hyperparameters.eta_min)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = WaymoH5Dataset(self.data_dir, self.hyperparameters.train_with_tracks_to_predict)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.hyperparameters.num_workers,
            shuffle=True,
            persistent_workers=True if not self.fast_dev_run else False,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=4 if not self.fast_dev_run else None,
            collate_fn=collate_waymo,
        )


def main(data_dir: str, fast_dev_run: bool, use_gpu: bool, ckpt_path: T.Optional[str]):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    hyperparameters = OmegaConf.load("configs/hyperparameters.yaml")
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

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="model-{epoch:02d}-{loss/train:.2f}",
        monitor="loss/train",
        mode="min",
        save_top_k=1,
        verbose=True,
    )
    metrics_callback = OnTrainCallback(data_dir, hyperparameters.train_with_tracks_to_predict)

    trainer = L.Trainer(
        max_epochs=hyperparameters.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        fast_dev_run=fast_dev_run,
        precision="16-mixed",
        callbacks=[metrics_callback, checkpoint_callback],
        accumulate_grad_batches=hyperparameters.accumulate_grad_batches,
        profiler=SimpleProfiler(),
        gradient_clip_val=hyperparameters.grad_norm_clip,
    )
    
    if LR_FIND and not fast_dev_run:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(module, min_lr=1e-6, max_lr=5e-2)

        fig = lr_finder.plot(suggest=True)
        fig.savefig("learning_rate.png")
        new_lr = lr_finder.suggestion()
        print("LEARNING RATE SUGGESTION: ", new_lr)
        assert False, "Terminate the program to avoid training"

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
