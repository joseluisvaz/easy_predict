import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import CrossAttentionBlock, SelfAttentionBlock
from models.polyline_encoder import PointNetPolylineEncoder, build_mlps
from models.rnn_cells import MultiAgentLSTMCell
from waymo_loader.feature_description import GT_STATES_MEANS, GT_STATES_STDS, ROADGRAPH_MEANS, ROADGRAPH_STDS


def concatenate_historical_features(history_states: torch.Tensor, actor_types: torch.Tensor):
    """Concatenate history states with one-hot encoded actor types"""
    n_batch, n_agents, n_past, _ = history_states.shape
    # clamp to 0, to remove instances of -1
    types = actor_types.clamp_(min=0).view(n_batch, n_agents, 1).expand(-1, -1, n_past)

    NUM_CLASSES: T.Final = 5
    types_one_hot = F.one_hot(types, num_classes=NUM_CLASSES).float()
    return torch.cat(
        [
            history_states,
            types_one_hot,
        ],
        axis=-1,
    )


def concatenate_map_features(map_feats: torch.Tensor, map_types: torch.Tensor):
    """Concatenate map_feats with one-hot encoded map types"""
    n_batch, n_polylines, n_points, _ = map_feats.shape
    types = map_types.view(n_batch, n_polylines, 1).expand(-1, -1, n_points)

    NUM_CLASSES: T.Final = 20
    types_one_hot = F.one_hot(types, num_classes=NUM_CLASSES).float()
    return torch.cat(
        [
            map_feats,
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
    INPUT_SIZE = 2  # x, y
    OUTPUT_SIZE = 2  # dx, dy

    def __init__(self, hidden_size, n_timesteps):
        super(Decoder, self).__init__()
        self.lstm_cell = MultiAgentLSTMCell(128, hidden_size)
        self.linear = nn.Linear(hidden_size, self.OUTPUT_SIZE)
        self.n_timesteps = n_timesteps
        self.hidden_size = hidden_size

        # Attention mechanisms
        self.polyline_interaction = CrossAttentionBlock(
            embed_dim=hidden_size, num_heads=8, dropout_p=0.0
        )

        self.state_encoder = build_mlps(
            self.INPUT_SIZE, [128, 64, 128], ret_before_act=True, without_norm=True
        )

    def forward(
        self, current_features, current_availabilities, hidden, context, map_embedding, map_avails
    ):

        current_state = current_features[..., :2]  # x, y

        outputs = []
        for t in range(self.n_timesteps):

            state_embedding = self.state_encoder(current_state)

            state_embedding = self.polyline_interaction(
                state_embedding, map_embedding, cross_mask=~map_avails
            )

            hidden, context = self.lstm_cell(
                state_embedding, (hidden, context), current_availabilities
            )

            current_state = current_state + self.linear(hidden)
            outputs.append(current_state)

        outputs = torch.stack(outputs, dim=2)
        return outputs


class PredictionModel(nn.Module):

    NUM_MCG_LAYERS = 4
    MAP_INPUT_SIZE = 20  # x, y, direction and one hot encoded type

    def __init__(self, input_features, hidden_size, n_timesteps, normalize: bool = False):
        super(PredictionModel, self).__init__()
        self.encoder = Encoder(input_features, hidden_size)
        self.decoder = Decoder(hidden_size, n_timesteps)

        # Map encoder, same parameters as in the MTR repository
        self.polyline_encoder = PointNetPolylineEncoder(
            24, 64, num_layers=5, num_pre_layers=3, out_channels=128
        )
     
        # Attention mechanisms
        self.polyline_interaction = CrossAttentionBlock(
            embed_dim=hidden_size, num_heads=8, dropout_p=0.0
        )
        self.actor_interaction = CrossAttentionBlock(
            embed_dim=hidden_size, num_heads=8, dropout_p=0.0
        )

        self.normalize = normalize

    def forward(
        self,
        history_states,
        history_availabilities,
        actor_types,
        roadgraph_features,
        roadgraph_features_mask,
        roadgraph_types,
    ):
        # map_feats.shape is (batch, polyline, points, features)
        map_feats = roadgraph_features
        map_types = roadgraph_types
        map_avails = roadgraph_features_mask

        history_features = concatenate_historical_features(history_states, actor_types)
        map_feats = concatenate_map_features(map_feats, map_types)
        current_features = history_features[:, :, -1].clone()
        current_availabilities = history_availabilities[:, :, -1]

        polyline_avails = torch.any(map_avails, dim=2)

        # Encode map into its own embedding
        hidden_polyline = self.polyline_encoder(map_feats, map_avails)  # [N_BATCH, N_POLYLINES]

        # Encode actors into their embedding using an RNN
        hidden_actors, context_actors = self.encoder(history_features, history_availabilities)
        
        # Attend actor to polylines, output size of actors
        attend_to_polyline = self.polyline_interaction(
            hidden_actors, hidden_polyline, cross_mask=~polyline_avails
        )

        # For now decoder only takes positions
        attend_to_actors = self.actor_interaction(
            hidden_actors,
            attend_to_polyline,
            cross_mask=~current_availabilities,
        )
        output = self.decoder(
            current_features,
            current_availabilities,
            attend_to_actors,
            context_actors,
            hidden_polyline,
            polyline_avails,
        )
        return output
