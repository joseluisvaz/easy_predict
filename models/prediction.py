import typing as T

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.multi_context_gating import MultiContextGating
from models.rnn_cells import MultiAgentLSTMCell
from models.multiheaded_attention import MultiheadAttention


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
        x = x.masked_fill(invalid, float("-inf"))
        x_pooled = x.amax(dim=2)
        x_pooled = x_pooled.masked_fill(invalid_polyline, 0.0)

        assert not torch.isnan(x_pooled).any()
        assert not torch.isinf(x_pooled).any()
        return x_pooled, polyline_map_avails


class PredictionModel(nn.Module):

    XY_OUTPUT_SIZE = 2
    NUM_MCG_LAYERS = 4
    MAP_INPUT_SIZE = 20  # x, y, direction and one hot encoded type

    def __init__(self, input_features, hidden_size, n_timesteps):
        super(PredictionModel, self).__init__()
        self.encoder = Encoder(input_features, hidden_size)
        self.decoder = Decoder(self.XY_OUTPUT_SIZE, hidden_size, self.XY_OUTPUT_SIZE, n_timesteps)

        # Map parameters
        self.mlp1 = nn.Linear(2, 128)
        self.polyline_encoder = PolylineEncoder(128, n_layer=3, mlp_dropout_p=0.0)
    
        # Attention mechanisms
        self.polyline_interaction = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=0.0, batch_first=True)
        self.actor_interaction = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=0.0, batch_first=True)

    def forward(self, inputs: T.Dict[str, torch.Tensor]):

        history_features = concatenate_historical_features(inputs)
        history_availabilities = inputs["history_availabilities"]

        # map_feats.shape is (batch, polyline, points, features)
        map_feats = torch.nested.to_padded_tensor(inputs["roadgraph_features"], padding=0.0)
        map_avails = torch.nested.to_padded_tensor(
            inputs["roadgraph_features_mask"], padding=False
        ).bool()[..., 0]

        polyline_avails = torch.any(map_avails, dim=2)
        map_invalid = ~map_avails
        pl_invalid = ~polyline_avails

        x_map = self.mlp1(map_feats)
        hidden_polyline = self.polyline_encoder(x_map, map_invalid)

        hidden_actors, context_actors = self.encoder(history_features, history_availabilities)

        attend_to_polyline, _ = self.polyline_interaction(hidden_actors, hidden_polyline, hidden_polyline, key_padding_mask=pl_invalid)
        attend_to_actors, _ = self.actor_interaction(hidden_actors, attend_to_polyline, attend_to_polyline, key_padding_mask=~current_availabilities)

        current_positions = history_features[:, :, -1, :2]  # For now decoder only takes positions
        current_availabilities = history_availabilities[:, :, -1]
        output = self.decoder(current_positions, current_availabilities, attend_to_actors, context_actors)
        return output
