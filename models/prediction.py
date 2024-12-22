import typing as T

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.multi_context_gating import MultiContextGating
from models.rnn_cells import MultiAgentLSTMCell
from models.multiheaded_attention import MultiheadAttention
from waymo_loader.feature_description import NUM_HISTORY_FRAMES


def seq_pooling(x: Tensor, invalid: Tensor) -> Tensor:
    """Do max pooling over the sequence dimension.
    Args:
        x: [n_sc, n_ag, n_step, hidden_dim] or [n_sc, n_mp, n_mp_pl_node, hidden_dim]
        invalid: [n_sc, n_ag, n_step]
    Returns:
        x_pooled: [n_sc, n_ag, hidden_dim]
    """
    x_pooled = x.masked_fill(invalid.unsqueeze(-1), float("-inf")).amax(2)
    return x_pooled.masked_fill(invalid.all(-1, keepdim=True), 0)


class MLP(nn.Module):
    def __init__(
        self,
        fc_dims: T.Union[T.List, T.Tuple],
        dropout_p: float = -1.0,
        end_layer_activation: bool = True,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        assert len(fc_dims) >= 2

        layers: T.List[nn.Module] = []
        for i in range(0, len(fc_dims) - 1):
            layers.append(nn.Linear(fc_dims[i], fc_dims[i + 1]))

            if (i < len(fc_dims) - 2) or (i == len(fc_dims) - 2 and end_layer_activation):
                if use_layernorm:
                    layers.append(nn.LayerNorm(fc_dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))

            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))

        self.input_dim = fc_dims[0]
        self.output_dim = fc_dims[-1]
        self.fc_layers = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        mask_invalid: T.Optional[torch.Tensor] = None,
        fill_invalid: float = 0.0,
    ) -> Tensor:
        """
        Args:
            x: [..., input_dim]
            mask_invalid: [...]
        Returns:
            x: [..., output_dim]
        """
        x = self.fc_layers(x.flatten(0, -2)).view(*x.shape[:-1], self.output_dim)
        if mask_invalid is not None:
            x = x.masked_fill(mask_invalid.unsqueeze(-1), fill_invalid)
        return x


class PolylineEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_layer: int = 3,
        mlp_use_layernorm: bool = False,
        mlp_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        mlp_layers: T.List[nn.Module] = []
        for _ in range(n_layer):
            mlp_layers.append(
                MLP(
                    [hidden_dim, hidden_dim // 2],
                    dropout_p=mlp_dropout_p,
                    use_layernorm=mlp_use_layernorm,
                    end_layer_activation=True,
                )
            )
        self.mlp_layers = nn.ModuleList(mlp_layers)

    def forward(self, x: Tensor, invalid: Tensor) -> Tensor:
        """c.f. VectorNet and SceneTransformer, Aggregate polyline/track level feature.
        Args:
            x: [n_sc, n_mp, n_mp_pl_node, hidden_dim]
            invalid: [n_sc, n_mp, n_mp_pl_node]
        Returns:
            emb: [n_sc, n_mp, hidden_dim]
        """
        _, _, n_mp_pl_node = invalid.shape

        for mlp in self.mlp_layers:
            x = mlp(x, invalid, float("-inf"))
            x = torch.cat((x, x.amax(dim=2, keepdim=True).expand(-1, -1, n_mp_pl_node, -1)), dim=-1)
            x.masked_fill_(invalid.unsqueeze(-1), 0)
        emb = seq_pooling(x, invalid)
        return emb


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
        self.lstm_cell = MultiAgentLSTMCell(self.INPUT_SIZE, hidden_size)
        self.linear = nn.Linear(hidden_size, self.OUTPUT_SIZE)
        self.n_timesteps = n_timesteps

    def forward(self, current_features, current_availabilities, hidden, context):

        current_state = current_features[..., :2]  # x, y

        outputs = []
        for t in range(self.n_timesteps):
            hidden, context = self.lstm_cell(
                current_state, (hidden, context), current_availabilities
            )
            delta_state = self.linear(hidden)
            current_state = current_state + delta_state
            outputs.append(current_state)

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
        x = x.masked_fill(invalid, float("-inf"))
        x_pooled = x.amax(dim=2)
        x_pooled = x_pooled.masked_fill(invalid_polyline, 0.0)

        assert not torch.isnan(x_pooled).any()
        assert not torch.isinf(x_pooled).any()
        return x_pooled, polyline_map_avails


class PredictionModel(nn.Module):

    NUM_MCG_LAYERS = 4
    MAP_INPUT_SIZE = 20  # x, y, direction and one hot encoded type

    def __init__(self, input_features, hidden_size, n_timesteps):
        super(PredictionModel, self).__init__()
        self.encoder = Encoder(input_features, hidden_size)
        self.decoder = Decoder(hidden_size, n_timesteps)

        # Map parameters
        self.polyline_adapter = nn.Linear(24, 128)
        self.polyline_encoder = PolylineEncoder(128, n_layer=3, mlp_dropout_p=0.0)

        # Attention mechanisms
        self.polyline_interaction = nn.MultiheadAttention(
            embed_dim=128, num_heads=8, dropout=0.0, batch_first=True
        )
        self.actor_interaction = nn.MultiheadAttention(
            embed_dim=128, num_heads=8, dropout=0.0, batch_first=True
        )

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

        polyline_avails = torch.any(map_avails, dim=2)
        map_invalid = ~map_avails
        pl_invalid = ~polyline_avails

        # Encode map
        x_map = self.polyline_adapter(map_feats)
        hidden_polyline = self.polyline_encoder(x_map, map_invalid)

        # Encode actors
        hidden_actors, context_actors = self.encoder(history_features, history_availabilities)

        # Attend actor to polylines
        attend_to_polyline, _ = self.polyline_interaction(
            hidden_actors, hidden_polyline, hidden_polyline, key_padding_mask=pl_invalid
        )

        # For now decoder only takes positions
        current_features = history_features[:, :, -1]
        current_availabilities = history_availabilities[:, :, -1]
        attend_to_actors, _ = self.actor_interaction(
            hidden_actors,
            attend_to_polyline,
            attend_to_polyline,
            key_padding_mask=~current_availabilities,
        )
        output = self.decoder(
            current_features, current_availabilities, attend_to_actors, context_actors
        )
        return output
