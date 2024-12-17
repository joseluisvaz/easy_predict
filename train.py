import io
from argparse import ArgumentParser, Namespace
import itertools
import warnings

# Ignore the warning about nested tensors to not spam the terminal
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.",
)

from PIL import Image
import torch

torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from clearml import Task
import lightning as L
from lightning.pytorch.tuner import Tuner

from waymo_loader.dataloaders import WaymoH5Dataset, collate_waymo
from models.prediction import PredictionModel
from waymo_loader.feature_description import _ROADGRAPH_IDX_TO_TYPE, _ROADGRAPH_TYPE_TO_COLOR

plt.style.use("dark_background")
torch.set_float32_matmul_precision("medium")


def compute_loss(predicted_positions, target_positions, target_availabilities):
    errors = torch.sum((target_positions - predicted_positions) ** 2, dim=-1)
    return torch.mean(errors * target_availabilities)


task = Task.init(project_name="TrajectoryPrediction", task_name="SimpleAgentPrediction MCG")


class OnTrainCallback(L.Callback):
    def __init__(self, dataset):
        super().__init__()

        self._n_samples = 5
        self._dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_waymo,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        for idx, single_sample_batch in enumerate(
            itertools.islice(self._dataloader, self._n_samples)
        ):
            # Just extract a single sample from the batch and keep the batch dimension
            self.log_plot_to_clearml(
                pl_module, task.get_logger(), single_sample_batch, trainer.current_epoch, idx
            )

    def log_plot_to_clearml(self, pl_module, logger, single_sample, current_epoch, scene_idx: int):
        """Creates the image of a single scenario and logs it to ClearML.

        Assumption: Each component of the scenario has the batch dimension of size 1
        """

        # Make sure the input is in the gpu
        single_sample = {k: v.cuda() for k, v in single_sample.items()}

        with torch.no_grad():
            predicted_positions = pl_module.model(single_sample)
            predicted_positions = predicted_positions[0].cpu().numpy()

        history_positions = single_sample["history_positions"][0].cpu().numpy()
        target_positions = single_sample["target_positions"][0].cpu().numpy()
        history_availabilities = single_sample["history_availabilities"][0].cpu().numpy()
        target_availabilities = single_sample["target_availabilities"][0].cpu().numpy()

        n_agents = history_positions.shape[0]

        past_sequences = []
        ground_truth_sequences = []
        predicted_sequences = []
        for agent_idx in range(n_agents):
            if history_availabilities[agent_idx, -1] == 0:
                # Agent is not available in the current timestep
                continue

            history_sequence = history_positions[agent_idx, history_availabilities[agent_idx], :2]

            ground_truth_sequence = target_positions[
                agent_idx, target_availabilities[agent_idx], :2
            ]
            # Use the entire prediction horizon
            predicted_sequence = predicted_positions[agent_idx, :, :2]

            past_sequences.append(history_sequence)
            ground_truth_sequences.append(ground_truth_sequence)
            predicted_sequences.append(predicted_sequence)

        map_features = torch.nested.to_padded_tensor(
            single_sample["roadgraph_features"], padding=0.0
        ).cpu()
        map_avails = torch.nested.to_padded_tensor(
            single_sample["roadgraph_features_mask"], padding=False
        ).cpu()
        map_types = torch.nested.to_padded_tensor(
            single_sample["roadgraph_features_types"], padding=False
        ).cpu()
        map_ids = torch.nested.to_padded_tensor(
            single_sample["roadgraph_features_ids"], padding=False
        ).cpu()

        map_features = map_features.view(-1, 2)
        map_avails = map_avails.view(-1, 1)[:, 0] # [N,]
        map_points = map_features[map_avails]  # [N, 2]
        map_types = map_types.view(-1)[map_avails].to(torch.int32)  # [N,]
        map_ids = map_ids.view(-1)[map_avails].to(torch.int32)  # [N,]

        plt.figure()

        unique_polyline_ids = torch.unique(map_ids)
        for polyline_id in unique_polyline_ids:
            filtered_points = map_points[map_ids == polyline_id]
            type_id = map_types[map_ids == polyline_id][0]
            color = _ROADGRAPH_TYPE_TO_COLOR[_ROADGRAPH_IDX_TO_TYPE[type_id.item()]]
            plt.plot(filtered_points[:, 0], filtered_points[:, 1], color=color, linewidth=0.5)

        # plt.scatter(map_points[:, 0], map_points[:, 1], s=0.05, c="gray")
        for sequence in past_sequences:
            plt.plot(sequence[:, 0], sequence[:, 1], color="gray", linewidth=5.0, alpha=0.7)
        for sequence in ground_truth_sequences:
            plt.plot(sequence[:, 0], sequence[:, 1], color="teal", linewidth=5.0, alpha=0.7)
        for sequence in predicted_sequences:
            plt.plot(sequence[:, 0], sequence[:, 1], color="orange", linewidth=5.0, alpha=0.7)

        plt.title("predictions")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)

        # Save the plot to an in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)

        # Convert the buffer to a PIL Image
        image = Image.open(buf)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        logger.report_image(
            title="sample", series=f"scene{scene_idx}", iteration=current_epoch, image=image
        )


class LightningModule(L.LightningModule):
    def __init__(self, dataset, fast_dev_run: bool):
        super().__init__()
        self.dataset = dataset
        self.fast_dev_run = fast_dev_run
        self.learning_rate = 0.002
        self.n_timesteps = 30
        self.batch_size = 32
        self.model = PredictionModel(
            input_features=13, hidden_size=128, n_timesteps=self.n_timesteps
        )

    def training_step(self, batch, batch_idx):
        predicted_positions = self.model(batch)

        # Crop the future positions to match the number of timesteps
        future_positions = batch["target_positions"][:, :, : self.n_timesteps]
        future_availabilities = batch["target_availabilities"][:, :, : self.n_timesteps]
        loss = compute_loss(predicted_positions, future_positions, future_availabilities)

        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predicted_positions = self.model(batch)

        # Crop the future positions to match the number of timesteps
        future_positions = batch["target_positions"][:, :, : self.n_timesteps]
        future_availabilities = batch["target_availabilities"][:, :, : self.n_timesteps]
        loss = compute_loss(predicted_positions, future_positions, future_availabilities)
        self.log("loss/val", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
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


def main(data_dir, fast_dev_run, use_gpu):
    dataset = WaymoH5Dataset(data_dir)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    module = LightningModule(dataset, fast_dev_run)
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        fast_dev_run=fast_dev_run,
        # precision="16-mixed",
        callbacks=OnTrainCallback(dataset),
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    main(args.data_dir, args.fast_dev_run, args.gpu)
