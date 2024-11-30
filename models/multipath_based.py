import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.multi_context_gating import MultiContextGating
from models.rnn_cells import MultiAgentLSTMCell
from models.vectornet import VectorNetMapEncoder
from models.multiheaded_attention import MultiheadAttention


def concatenate_historical_features(batch):
    n_batch, n_agents, n_past, _ = batch["history_positions"].shape
    types = batch["actor_type"].view(n_batch, n_agents, 1)

    NUM_CLASSES: T.Final = 5
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


class MapPointNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MapPointNet, self).__init__()
        self.mlp1 = nn.Linear(input_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, output_size)

    def forward(self, map_feats, map_avails):
        polyline_map_avails = torch.any(map_avails, dim=2)
        invalid = ~map_avails
        invalid_polyline = ~polyline_map_avails

        x = F.relu(self.mlp1(map_feats))
        x = self.mlp2(x)

        # Mask to have -inf so that we can do the max pooling
        x.masked_fill_(invalid, float("-inf"))
        x_pooled = x.amax(dim=2)
        x_pooled.masked_fill_(invalid_polyline, float("-inf"))
        return x_pooled, polyline_map_avails


class MultiPathBased(nn.Module):

    XY_OUTPUT_SIZE = 2
    NUM_MCG_LAYERS = 4
    MAP_INPUT_SIZE = 20  # x, y, direction and one hot encoded type

    def __init__(self, input_features, hidden_size, n_timesteps):
        super(MultiPathBased, self).__init__()
        self.encoder = Encoder(input_features, hidden_size)
        self.mcg = MultiContextGating(hidden_size, n_contexts=self.NUM_MCG_LAYERS)
        self.decoder = Decoder(self.XY_OUTPUT_SIZE, hidden_size, self.XY_OUTPUT_SIZE, n_timesteps)
        self.actor_interaction = MultiheadAttention(
            in_feats=hidden_size, per_head_feats=16, n_heads=16, dropout_factor=0.0
        )
        self.map_encoder = MapPointNet(2, 64, 128)
        self.polyline_interaction = MultiheadAttention(
            in_feats=128, per_head_feats=16, n_heads=16, dropout_factor=0.0
        )

    def forward(self, inputs: T.Dict[str, torch.Tensor]):

        history_features = concatenate_historical_features(inputs)
        history_availabilities = inputs["history_availabilities"]

        # map_feats.shape is (batch, polyline, points, features)
        map_feats = torch.nested.to_padded_tensor(inputs["roadgraph_features"], padding=0.0)
        map_avails = torch.nested.to_padded_tensor(
            inputs["roadgraph_features_mask"], padding=False
        ).bool()

        polyline_hidden, polyline_avails = self.map_encoder(map_feats, map_avails)

        hidden, context = self.encoder(history_features, history_availabilities)
        current_positions = history_features[:, :, -1, :2]  # For now only takes positions
        current_availabilities = history_availabilities[:, :, -1]

        hidden, _ = self.polyline_interaction(
            hidden, polyline_hidden, polyline_hidden, mask=polyline_avails
        )
        hidden, _ = self.actor_interaction(hidden, hidden, hidden, mask=current_availabilities)

        output = self.decoder(current_positions, current_availabilities, hidden, context)
        return output
