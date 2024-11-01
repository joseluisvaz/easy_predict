from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from clearml import Task
import lightning as L

from waymo_loader.dataloaders import  WaymoH5Dataset, collate_waymo
from models.rnn_cells import MultiAgentLSTMCell


def concatenate_historical_features(batch):
    return torch.cat(
        [ 
            batch["history_positions"],
            batch["history_velocities"],
            batch["history_yaws"],
        ], axis=-1   
    )

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm_cell = MultiAgentLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, sequence, mask):
        n_batch, n_agents, n_timesteps, _ = sequence.shape
        hidden, context = self.lstm_cell.get_initial_hidden_state((n_batch, n_agents, self.hidden_size), sequence.device)
        
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
            output = self.linear(hidden)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=2)
        return outputs 

class SimpleAgentPrediction(nn.Module):

    XY_OUTPUT_SIZE = 2

    def __init__(self, input_features, hidden_size, n_timesteps):
        super(SimpleAgentPrediction, self).__init__()
        self.encoder = Encoder(input_features, hidden_size)
        self.decoder = Decoder(self.XY_OUTPUT_SIZE, hidden_size, self.XY_OUTPUT_SIZE, n_timesteps)

    def forward(self, history_features, history_availabilities):
        hidden, context = self.encoder(history_features, history_availabilities)
        
        current_positions = history_features[:, :, -1, :2] # For now only takes positions
        current_availabilities = history_availabilities[:, :, -1]
        output = self.decoder(current_positions, 
                              current_availabilities, 
                              hidden, 
                              context)
        return output

def compute_loss(predicted_positions, target_positions, target_availabilities):
    errors = torch.sum((target_positions - predicted_positions) ** 2, dim=-1)
    return torch.mean(errors * target_availabilities)


class LightningModule(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.n_timesteps = 30
            self.model = SimpleAgentPrediction(input_features=5, hidden_size=128, n_timesteps=self.n_timesteps)

        def training_step(self, batch, batch_idx):
            historical_features = concatenate_historical_features(batch)
            predicted_positions = self.model(historical_features, batch["history_availabilities"])

            # Crop the future positions to match the number of timesteps
            future_positions = batch["target_positions"][:, :, :self.n_timesteps]
            future_availabilities = batch["target_availabilities"][:, :, :self.n_timesteps]
            loss = compute_loss(predicted_positions, future_positions, future_availabilities)

            self.log("loss/train", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            historical_features = concatenate_historical_features(batch)
            predicted_positions = self.model(historical_features, batch["history_availabilities"])
            
            # Crop the future positions to match the number of timesteps
            future_positions = batch["target_positions"][:, :, :self.n_timesteps]
            future_availabilities = batch["target_availabilities"][:, :, :self.n_timesteps]
            loss = compute_loss(predicted_positions, future_positions, future_availabilities)
            self.log("loss/val", loss)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

task = Task.init(project_name="TrajectoryPrediction", task_name="SimpleAgentPrediction") 

def main(data_dir, fast_dev_run):
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

    module = LightningModule()
    trainer = L.Trainer(max_epochs=100, accelerator="gpu", devices=1, fast_dev_run=fast_dev_run)
    trainer.fit(model=module, train_dataloaders=train_loader)
   

def _parse_arguments() -> Namespace:
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the folder with the tf records.")
    parser.add_argument("--fast-dev-run", action="store_true", help="Path to the folder with the tf records.")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_arguments()
    main(args.data_dir, args.fast_dev_run)