import typing as T

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data_utils.feature_description import (
    NUM_FUTURE_FRAMES,
    NUM_HISTORY_FRAMES,
)
from data_utils.feature_generation import collate_waymo_scenario, collate_waymo_stack
from data_utils.processed_dataset import AgentCentricDataset, ScenarioDataset
from common_utils.metrics import MotionMetrics, _default_metrics_config
from models.inference import run_model_forward_pass
from models.prediction import PredictionModel


def compute_loss(
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor,
    target_availabilities: torch.Tensor,
    agent_mask: torch.Tensor,
) -> float:
    """
    B: batch size
    N: number of agents
    T: number of time steps

    Args:
        predicted_positions (torch.Tensor): [B, N, T, 2]
        target_positions (torch.Tensor): [B, N, T, 2]
        target_availabilities (torch.Tensor): [B, N, T]
        agent_mask (torch.Tensor): [B, N]
    """

    total_mask = target_availabilities
    total_mask *= agent_mask.unsqueeze(-1)

    # Sum over positions dimension
    errors = torch.sum((target_positions - predicted_positions) ** 2, dim=-1)
    return torch.mean(errors * total_mask)


AGENT_INPUT_FEATURES: T.Final[int] = 12


class PredictionLightningModule(L.LightningModule):
    def __init__(
        self,
        fast_dev_run: bool,
        hyperparameters: DictConfig,
        clearml_task: T.Optional[T.Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hp = hyperparameters

        self.task = clearml_task
        self.fast_dev_run = fast_dev_run

        self.model = PredictionModel(
            input_features=AGENT_INPUT_FEATURES,
            hidden_size=hyperparameters.hidden_size,
            n_timesteps=NUM_FUTURE_FRAMES,
            model_config=hyperparameters.model_config,
        )

        # self.model = torch.compile(self.model)
        self.metrics = MotionMetrics(_default_metrics_config())

    def _inference_and_loss(self, batch):
        predicted_features = run_model_forward_pass(self.model, batch)
        predicted_positions = predicted_features[..., :2]

        batch_indices = torch.arange(
            predicted_positions.shape[0], device=predicted_positions.device
        )
        agent_to_predict = batch["agent_to_predict"]

        # Crop the future positions to match the number of timesteps
        future_positions = batch["gt_features"][
            batch_indices, agent_to_predict, -NUM_FUTURE_FRAMES:, :2
        ][:, None]
        future_availabilities = batch["gt_features_avails"][
            batch_indices, agent_to_predict, -NUM_FUTURE_FRAMES:
        ][:, None]
        current_availabilities = batch["gt_features_avails"][
            batch_indices, agent_to_predict, NUM_HISTORY_FRAMES
        ][:, None]

        assert torch.all(current_availabilities)

        loss = compute_loss(
            predicted_positions,
            future_positions,
            future_availabilities,
            current_availabilities,
        )
        return predicted_positions, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._inference_and_loss(batch)
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            predicted_positions, loss = self._inference_and_loss(batch)
            self.metrics.update_state(batch, predicted_positions)
            self.log("loss/val", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hp.learning_rate, weight_decay=self.hp.weight_decay
        )

        cosine_t_max = self.hp.max_epochs * len(self.train_dataloader())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/train",
                "interval": "step",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        dataset = AgentCentricDataset(self.hp.train_dataset)
        return DataLoader(
            dataset,
            batch_size=self.hp.batch_size,
            num_workers=self.hp.num_workers,
            shuffle=True,
            persistent_workers=True if not self.fast_dev_run else False,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=4 if not self.fast_dev_run else None,
            collate_fn=collate_waymo_stack,
        )

    def val_dataloader(self):
        dataset = ScenarioDataset(self.hp.val_dataset)
        return DataLoader(
            dataset,
            batch_size=self.hp.val_batch_size,
            num_workers=self.hp.num_workers,
            shuffle=False,
            persistent_workers=True if not self.fast_dev_run else False,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=4 if not self.fast_dev_run else None,
            collate_fn=collate_waymo_scenario,
        )
