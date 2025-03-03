import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.attention import CrossAttentionBlock, SelfAttentionBlock
from models.modules import DynamicsLayer
from models.pointnet_encoder import PointNetPolylineEncoder, build_mlps
from models.rnn_cells import MultiAgentLSTMCell

MAX_NUM_TRACKS_TO_PREDICT: T.Final = 8


def concatenate_historical_features(
    history_states: torch.Tensor, actor_types: torch.Tensor
) -> torch.Tensor:
    """Concatenate history states with one-hot encoded actor types.
    Args:
        history_states: (n_batch, n_agents, n_past, _)
        actor_types: (n_batch, n_agents, n_past)
    Returns:
        (n_batch, n_agents, n_past, n_features + n_classes)
    """
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


def concatenate_map_features(
    map_feats: torch.Tensor, map_types: torch.Tensor
) -> torch.Tensor:
    """Concatenate map_feats with one-hot encoded map types.
    Args:
        map_feats: (n_batch, n_polylines, n_points, _)
        map_types: (n_batch, n_polylines, n_points)
    Returns:
        (n_batch, n_polylines, n_points, n_features + n_classes)
    """
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


def concatenate_tl_features(
    tl_feats: torch.Tensor, tl_types: torch.Tensor
) -> torch.Tensor:
    """Concatenate map_feats with one-hot encoded map types.
    Args:
        tl_feats: (n_batch, n_tls, n_timesteps, _)
        tl_types: (n_batch, n_tls, n_timesteps)
    Returns:
        (n_batch, n_tls, n_timesteps, n_features + n_classes)
    """

    NUM_CLASSES: T.Final = 9
    types_one_hot = F.one_hot(tl_types, num_classes=NUM_CLASSES).float()
    return torch.cat(
        [
            tl_feats,
            types_one_hot,
        ],
        axis=-1,
    )


class Decoder(nn.Module):
    def __init__(self, n_timesteps: int, config: DictConfig):
        super(Decoder, self).__init__()

        self.hidden_size = config.hidden_size
        self.config = config.decoder
        self.actor_input_size = config.actor_input_size

        self.lstm_cell = MultiAgentLSTMCell(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.config.output_size)
        self.n_timesteps = n_timesteps
        self.dynamics = DynamicsLayer(
            self.config.dynamics_layer.delta_t,
            self.config.dynamics_layer.max_acc,
            self.config.dynamics_layer.max_yaw_rate,
        )

        # Attention mechanism to attend to the context vectors
        self.context_attention = CrossAttentionBlock(
            embed_dim=self.hidden_size,
            num_heads=self.config.cross_attention.n_heads,
            dropout_p=self.config.cross_attention.dropout_p,
        )

        self.state_encoder = build_mlps(
            self.actor_input_size,
            [self.hidden_size, self.hidden_size // 2, self.hidden_size],
            ret_before_act=True,
            without_norm=True,
        )

    def step_physical_state(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Update the feature vector using the dynamics function, this copies
        all the features to the policy and replaces the state with the new computed states
        Args:
            state: (n_batch, n_agents, n_features)
            action: (n_batch, n_agents, n_actions)
        Returns:
            next_state: (n_batch, n_agents, n_features)
        """
        next_state = state.clone()
        physical_state = self.dynamics.get_state_from_features(state)
        next_physical_state = self.dynamics(physical_state, action)
        next_state[..., : next_physical_state.shape[-1]] = next_physical_state
        return next_state

    def forward(
        self,
        current_features: torch.Tensor,
        current_availabilities: torch.Tensor,
        context_embeddings: torch.Tensor,
        context_avails: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the decoder. when we are using agent centric representation we are only
        decoding a single agent, therefore n_agents is 1.

        Args:
            current_features: (n_batch, n_agents, n_features)
            current_availabilities: (n_batch, n_agents)
            context_embeddings: (n_batch, n_entities, _)
            context_avails: (n_batch, n_entities)
        Returns:
            outputs: (n_batch, n_agents, n_timesteps, n_features)
        """
        current_state = current_features.clone()

        n_batch, n_agents, _ = current_state.shape
        h = torch.zeros(n_batch, n_agents, self.hidden_size).to(current_state.device)
        c = torch.zeros(n_batch, n_agents, self.hidden_size).to(current_state.device)

        outputs = []
        for t in range(self.n_timesteps):
            state_embedding = self.state_encoder(current_state)

            state_embedding = self.context_attention(
                state_embedding, context_embeddings, cross_mask=~context_avails
            )

            h, c = self.lstm_cell(state_embedding, (h, c), current_availabilities)

            action = self.linear(h)
            current_state = self.step_physical_state(current_state, action)

            outputs.append(current_state)

        outputs = torch.stack(outputs, dim=2)
        return outputs


class MultiModalEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super(MultiModalEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.config = config.encoder
        self.actor_input_size = config.actor_input_size
        # Map encoder, same parameters as in the MTR repository
        self.actor_encoder = PointNetPolylineEncoder(
            self.actor_input_size,
            self.hidden_size // 2,
            num_layers=5,
            num_pre_layers=3,
            out_channels=self.hidden_size,
        )
        self.map_encoder = PointNetPolylineEncoder(
            self.config.map_input_size,
            self.hidden_size // 2,
            num_layers=self.config.point_net.num_layers,
            num_pre_layers=self.config.point_net.num_pre_layers,
            out_channels=self.hidden_size,
        )

        self.tl_encoder = PointNetPolylineEncoder(
            self.config.tl_input_size,
            self.hidden_size // 2,
            num_layers=self.config.point_net.num_layers,
            num_pre_layers=self.config.point_net.num_pre_layers,
            out_channels=self.hidden_size,
        )

    def forward(
        self,
        history_features: torch.Tensor,
        history_availabilities: torch.Tensor,
        roadgraph_features: torch.Tensor,
        roadgraph_availabilities: torch.Tensor,
        tl_features: torch.Tensor,
        tl_availabilities: torch.Tensor,
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        hidden_actor = self.actor_encoder(history_features, history_availabilities)
        hidden_map = self.map_encoder(roadgraph_features, roadgraph_availabilities)
        hidden_tl = self.tl_encoder(tl_features, tl_availabilities)

        # Check if there is any availability per entity, these are the valid entitites
        # that should be attended to even if they are not present for the entire sequence
        # length.
        actor_avails = torch.any(history_availabilities, dim=2)
        polyline_avails = torch.any(roadgraph_availabilities, dim=2)
        tl_persistent_avails = torch.any(tl_availabilities, dim=2)
        global_features = torch.cat([hidden_actor, hidden_map, hidden_tl], dim=1)
        global_avails = torch.cat(
            [actor_avails, polyline_avails, tl_persistent_avails], dim=1
        )

        return global_features, global_avails


class PredictionModel(nn.Module):
    def __init__(
        self,
        n_timesteps: int,
        config: DictConfig,
    ):
        super(PredictionModel, self).__init__()
        self.hidden_size = config.hidden_size
        self.decoder = Decoder(n_timesteps, config)
        self.encoder = MultiModalEncoder(config)

        self.global_attention = SelfAttentionBlock(
            embed_dim=self.hidden_size,
            num_heads=config.self_attention.n_heads,
            dropout_p=config.self_attention.dropout_p,
        )

    def forward(
        self,
        history_states: torch.Tensor,
        history_availabilities: torch.Tensor,
        actor_types: torch.Tensor,
        roadgraph_features: torch.Tensor,
        roadgraph_features_mask: torch.Tensor,
        roadgraph_types: torch.Tensor,
        tl_states: torch.Tensor,
        tl_states_categorical: torch.Tensor,
        tl_avails: torch.Tensor,
        agent_to_predict: torch.Tensor,
    ) -> torch.Tensor:
        history_features_ohe = concatenate_historical_features(
            history_states, actor_types
        )
        roadgraph_features_ohe = concatenate_map_features(
            roadgraph_features, roadgraph_types
        )
        tl_features_ohe = concatenate_tl_features(tl_states, tl_states_categorical)

        global_features, global_avails = self.encoder(
            history_features_ohe,
            history_availabilities,
            roadgraph_features_ohe,
            roadgraph_features_mask,
            tl_features_ohe,
            tl_avails,
        )

        # Perform self attention on all the features before sending it to the cross attention mask
        global_features = self.global_attention(global_features, mask=~global_avails)

        # Get the agent of interest to predict and add a new agent dimension of size 1 for compatibility
        # with the decoder
        batch_size = history_features_ohe.shape[0]
        batch_indices = torch.arange(batch_size, device=history_features_ohe.device)

        # Get the last history state of the agent of interest and index for the
        # agent to predict for each sample in the batch.
        current_features = history_features_ohe[batch_indices, agent_to_predict, -1][
            :, None
        ]

        # Get the last history availability of the agent of interest
        current_availabilities = history_availabilities[
            batch_indices, agent_to_predict, -1
        ][:, None]

        assert torch.all(current_availabilities), (
            "All current availabilities should be True "
        )

        output = self.decoder(
            current_features,
            current_availabilities,
            global_features,
            global_avails,
        )
        return output
