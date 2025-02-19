import typing as T

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

from data_utils.feature_description import NUM_FUTURE_FRAMES, NUM_HISTORY_FRAMES
from data_utils.feature_generation import collate_waymo_stack
from data_utils.processed_dataset import ProcessedDataset, AgentCentricDataset
from metrics import MotionMetrics, _default_metrics_config
from models.inference import run_model_forward_pass
from models.prediction import PredictionModel


def compute_loss(
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor,
    target_availabilities: torch.Tensor,
    agent_mask: torch.Tensor,
    loss_tracks_to_predict_mask: bool,
    loss_use_ego_vehicle_mask: bool,
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

    # Use masks to hide agents out
    # if loss_use_ego_vehicle_mask and loss_tracks_to_predict_mask:
    #     total_mask *= tracks_to_predict_mask.unsqueeze(-1) | is_sdc.unsqueeze(-1)
    # elif loss_use_ego_vehicle_mask:
    #     total_mask *= is_sdc.unsqueeze(-1)
    # elif loss_tracks_to_predict_mask:
    #     total_mask *= tracks_to_predict_mask.unsqueeze(-1)

    # Sum over positions dimension
    errors = torch.sum((target_positions - predicted_positions) ** 2, dim=-1)
    return torch.mean(errors * total_mask)


MAX_PREDICT_AGENTS: T.Final[int] = 8
AGENT_INPUT_FEATURES: T.Final[int] = 12


class LightningModule(L.LightningModule):
    def __init__(
        self,
        fast_dev_run: bool,
        hyperparameters: DictConfig,
        cosine_t_max: int,
        clearml_task: T.Optional[T.Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cosine_t_max = cosine_t_max
        self.task = clearml_task
        self.fast_dev_run = fast_dev_run
        self.learning_rate = hyperparameters.learning_rate
        self.n_timesteps = NUM_FUTURE_FRAMES
        self.batch_size = hyperparameters.batch_size
        self.model = PredictionModel(
            input_features=AGENT_INPUT_FEATURES,
            hidden_size=hyperparameters.hidden_size,
            n_timesteps=self.n_timesteps,
            model_config=hyperparameters.model_config,
            normalize=hyperparameters.normalize_features,
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
            object_type=actor_type,
        )

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

        # batch["tracks_to_predict"][:, :MAX_NUM_TRACKS_TO_PREDICT],
        # batch["is_sdc"][:, :MAX_NUM_TRACKS_TO_PREDICT],
        loss = compute_loss(
            predicted_positions,
            future_positions,
            future_availabilities,
            current_availabilities,
            self.hyperparameters.loss_tracks_to_predict_mask,
            self.hyperparameters.loss_use_ego_vehicle_mask,
        )
        return predicted_positions, loss

    def training_step(self, batch, batch_idx):

        _, loss = self._inference_and_loss(batch)

        # Compute the percentarge of elements in the map that are available, this ia metric that tells us
        # how empty the tensors are
        map_avails = batch["roadgraph_features_mask"]

        _, n_polyline, _ = map_avails.shape
        self.log("map_sizes/polyline", n_polyline)

        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log("lr", lr)

        percentage = map_avails.sum().float() / map_avails.numel()
        self.log("map_availability_percentage", percentage)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            predicted_positions, loss = self._inference_and_loss(batch)
            self._update_metrics(batch, predicted_positions)
            self.log("loss/val", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.hyperparameters.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cosine_t_max)
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
        dataset = AgentCentricDataset(
            self.hyperparameters.train_dataset,
            data_perturb_cfg=self.hyperparameters.data_perturb,
            # train_with_tracks_to_predict=self.hyperparameters.train_with_tracks_to_predict,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.hyperparameters.num_workers,
            shuffle=True,
            persistent_workers=True if not self.fast_dev_run else False,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=4 if not self.fast_dev_run else None,
            collate_fn=collate_waymo_stack,
        )

    def val_dataloader(self):
        dataset = AgentCentricDataset(
            self.hyperparameters.val_dataset,
            # train_with_tracks_to_predict=self.hyperparameters.train_with_tracks_to_predict,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.hyperparameters.num_workers,
            shuffle=False,
            persistent_workers=True if not self.fast_dev_run else False,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=4 if not self.fast_dev_run else None,
            collate_fn=collate_waymo_stack,
        )
