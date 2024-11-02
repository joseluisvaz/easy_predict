import io
from argparse import ArgumentParser, Namespace

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from clearml import Task
import lightning as L

from waymo_loader.dataloaders import WaymoH5Dataset, collate_waymo
from models.rnn_cells import MultiAgentLSTMCell
from models.multi_context_gating import MultiContextGating

torch.set_float32_matmul_precision("medium")

NUM_CLASSES = 5


def concatenate_historical_features(batch):
    n_batch, n_agents, n_past, _ = batch["history_positions"].shape
    types = batch["actor_type"].view(n_batch, n_agents, 1)
    types_one_hot = F.one_hot(types.expand(-1, -1, n_past), num_classes=NUM_CLASSES).float()

    # Add time dimension to the extent tensor and expand it to match the number of past timesteps
    extent = batch["extent"].view(n_batch, n_agents, 1, -1).expand(-1, -1, n_past, -1)

    return torch.cat(
        [
            batch["history_positions"],
            batch["history_velocities"],
            batch["history_yaws"],
            extent,
            types_one_hot,
        ],
        axis=-1,
    )


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm_cell = MultiAgentLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, sequence, mask):
        n_batch, n_agents, n_timesteps, _ = sequence.shape
        hidden, context = self.lstm_cell.get_initial_hidden_state(
            (n_batch, n_agents, self.hidden_size), sequence.device, requires_grad=True
        )

        for t in range(n_timesteps):
            input_t = sequence[:, :, t, :]
            mask_t = mask[:, :, t]
            hidden, context = self.lstm_cell(input_t, (hidden, context), mask_t)

        return hidden, context


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_timesteps):
        super(Decoder, self).__init__()
        self.lstm_cell = MultiAgentLSTMCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.n_timesteps = n_timesteps

    def forward(self, current_positions, current_availabilities, hidden, context):
        output = current_positions

        outputs = []
        for t in range(self.n_timesteps):
            hidden, context = self.lstm_cell(output, (hidden, context), current_availabilities)
            delta_output = self.linear(hidden)
            output = output + delta_output
            outputs.append(output)

        outputs = torch.stack(outputs, dim=2)
        return outputs


class SimpleAgentPrediction(nn.Module):

    XY_OUTPUT_SIZE = 2
    NUM_MCG_LAYERS = 4

    def __init__(self, input_features, hidden_size, n_timesteps):
        super(SimpleAgentPrediction, self).__init__()
        self.encoder = Encoder(input_features, hidden_size)
        self.mcg = MultiContextGating(hidden_size, n_contexts=self.NUM_MCG_LAYERS)
        self.decoder = Decoder(self.XY_OUTPUT_SIZE, hidden_size, self.XY_OUTPUT_SIZE, n_timesteps)

    def forward(self, history_features, history_availabilities):
        hidden, context = self.encoder(history_features, history_availabilities)
        current_positions = history_features[:, :, -1, :2]  # For now only takes positions
        current_availabilities = history_availabilities[:, :, -1]

        hidden = self.mcg(hidden, current_availabilities)

        output = self.decoder(current_positions, current_availabilities, hidden, context)
        return output


def compute_loss(predicted_positions, target_positions, target_availabilities):
    errors = torch.sum((target_positions - predicted_positions) ** 2, dim=-1)
    return torch.mean(errors * target_availabilities)


task = Task.init(project_name="TrajectoryPrediction", task_name="SimpleAgentPrediction MCG")


class OnTrainCallback(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # do something with all training_step outputs, for example:
        batch = next(iter(trainer.val_dataloaders))

        SAMPLE_IMAGES = 5
        n_batch = batch["history_positions"].shape[0]
        for idx in range(0, min(SAMPLE_IMAGES, n_batch)):
            # Just extract a single sample from the batch and keep the batch dimension
            single_sample = {k: v[idx].unsqueeze(0) for k, v in batch.items()}
            self.log_plot_to_clearml(
                pl_module, task.get_logger(), single_sample, trainer.current_epoch, idx
            )

    def log_plot_to_clearml(self, pl_module, logger, single_sample, current_epoch, scene_idx: int):
        """Creates the image of a single scenario and logs it to ClearML.

        Assumption: Each component of the scenario has the batch dimension of size 1
        """

        with torch.no_grad():
            historical_features = concatenate_historical_features(single_sample).to(
                pl_module.device
            )
            predicted_positions = pl_module.model(
                historical_features, single_sample["history_availabilities"].to(pl_module.device)
            )
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

        plt.figure()
        for sequence in past_sequences:
            plt.plot(sequence[:, 0], sequence[:, 1], "o-", color="gray")
        for sequence in ground_truth_sequences:
            plt.plot(sequence[:, 0], sequence[:, 1], "o-", color="green")
        for sequence in predicted_sequences:
            plt.plot(sequence[:, 0], sequence[:, 1], "o-", color="red")

        plt.title("predictions")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        # Save the plot to an in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
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
    def __init__(self):
        super().__init__()
        self.n_timesteps = 30
        self.model = SimpleAgentPrediction(
            input_features=13, hidden_size=128, n_timesteps=self.n_timesteps
        )

    def training_step(self, batch, batch_idx):
        historical_features = concatenate_historical_features(batch)
        predicted_positions = self.model(historical_features, batch["history_availabilities"])

        # Crop the future positions to match the number of timesteps
        future_positions = batch["target_positions"][:, :, : self.n_timesteps]
        future_availabilities = batch["target_availabilities"][:, :, : self.n_timesteps]
        loss = compute_loss(predicted_positions, future_positions, future_availabilities)

        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        historical_features = concatenate_historical_features(batch)
        predicted_positions = self.model(historical_features, batch["history_availabilities"])

        # Crop the future positions to match the number of timesteps
        future_positions = batch["target_positions"][:, :, : self.n_timesteps]
        future_availabilities = batch["target_availabilities"][:, :, : self.n_timesteps]
        loss = compute_loss(predicted_positions, future_positions, future_availabilities)
        self.log("loss/val", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main(data_dir, fast_dev_run, use_gpu):
    dataset = WaymoH5Dataset(data_dir)

    train_loader = DataLoader(
        dataset,
        batch_size=512,
        num_workers=8,
        shuffle=True,
        persistent_workers=True,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=8,
        collate_fn=collate_waymo,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=512,
        num_workers=8,
        shuffle=False,
        persistent_workers=True,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=8,
        collate_fn=collate_waymo,
    )

    module = LightningModule()
    trainer = L.Trainer(
        max_epochs=30,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        fast_dev_run=fast_dev_run,
        callbacks=OnTrainCallback(),
    )
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)


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
