import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import CrossAttentionBlock, SelfAttentionBlock
from models.modules import DynamicsLayer
from models.polyline_encoder import PointNetPolylineEncoder, build_mlps
from models.position_embeddings import gen_sineembed_for_position
from models.rnn_cells import MultiAgentLSTMCell

MAX_NUM_TRACKS_TO_PREDICT: T.Final = 8


def concatenate_historical_features(history_states: torch.Tensor, actor_types: torch.Tensor):
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


def concatenate_map_features(map_feats: torch.Tensor, map_types: torch.Tensor):
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

def concatenate_tl_features(tl_feats: torch.Tensor, tl_types: torch.Tensor):
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


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm_cell = MultiAgentLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, sequence, mask):
        """Encode the sequence using an LSTM cell.
        Args:
            sequence: (n_batch, n_agents, n_timesteps, _)
            mask: (n_batch, n_agents, n_timesteps)
        Returns:
            hidden: (n_batch, n_agents, n_timesteps, _)
            context: (n_batch, n_agents, n_timesteps, _)
        """
        
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
    def __init__(self, hidden_size, n_timesteps, config):
        super(Decoder, self).__init__()
        self.lstm_cell = MultiAgentLSTMCell(128, hidden_size)
        self.linear = nn.Linear(hidden_size, config.decoder.output_size)
        self.n_timesteps = n_timesteps
        self.hidden_size = hidden_size
        self.dynamics = DynamicsLayer(
            config.dynamics_layer.delta_t,
            config.dynamics_layer.max_acc,
            config.dynamics_layer.max_yaw_rate,
        )

        # Attention mechanism to attend to the context vectors
        self.context_attention = CrossAttentionBlock(
            embed_dim=hidden_size, num_heads=8, dropout_p=0.0
        )

        self.state_encoder = build_mlps(
            config.decoder.input_size, [128, 64, 128], ret_before_act=True, without_norm=True
        )

    def step_physical_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
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

    def forward(self, current_features, current_availabilities, context_embeddings, context_avails):
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


class PredictionModel(nn.Module):

    NUM_MCG_LAYERS = 4
    MAP_INPUT_SIZE = 20  # x, y, direction and one hot encoded type

    def __init__(
        self, input_features, hidden_size, n_timesteps, model_config,
    ):
        super(PredictionModel, self).__init__()
        self.hidden_size = hidden_size
        # self.encoder = Encoder(input_features, hidden_size)
        self.decoder = Decoder(hidden_size, n_timesteps, model_config)

        # Map encoder, same parameters as in the MTR repository
        self.actor_encoder = PointNetPolylineEncoder(
            input_features, 64, num_layers=5, num_pre_layers=3, out_channels=hidden_size
        )
        self.map_encoder = PointNetPolylineEncoder(
            24, 64, num_layers=5, num_pre_layers=3, out_channels=hidden_size
        )
        
        self.use_tl_encoder = model_config.tl_encoder.use_tl_encoder

        self.tl_encoder = PointNetPolylineEncoder(
            model_config.tl_encoder.input_size, 64, num_layers=5, num_pre_layers=3, out_channels=hidden_size
        ) if self.use_tl_encoder else None

        self.global_attention = SelfAttentionBlock(
            embed_dim=hidden_size, num_heads=8, dropout_p=0.0
        )


    def forward(
        self,
        history_states,
        history_availabilities,
        actor_types,
        roadgraph_features,
        roadgraph_features_mask,
        roadgraph_types,
        tl_states,
        tl_states_categorical,
        tl_avails,
        tracks_to_predict,
        agent_to_predict,
    ):
        # map_feats.shape is (batch, polyline, points, features)
        map_feats = roadgraph_features
        map_types = roadgraph_types
        map_avails = roadgraph_features_mask

        history_features = concatenate_historical_features(history_states, actor_types)
        map_feats = concatenate_map_features(map_feats, map_types)
        tl_feats = concatenate_tl_features(tl_states, tl_states_categorical)

        actor_avails = torch.any(history_availabilities, dim=2)
        polyline_avails = torch.any(map_avails, dim=2)

        hidden_actor = self.actor_encoder(history_features, history_availabilities)
        hidden_map = self.map_encoder(map_feats, map_avails)  # [N_BATCH, N_POLYLINES]
        
        if self.use_tl_encoder:
            tl_persistent_avails = torch.any(tl_avails, dim=2)
            hidden_tl = self.tl_encoder(tl_feats, tl_avails)  # [N_BATCH, N_TRAFFIC_LIGHTS]
            global_features = torch.cat([hidden_actor, hidden_map, hidden_tl], dim=1)
            global_avails = torch.cat([actor_avails, polyline_avails, tl_persistent_avails], dim=1)
        else:
            global_features = torch.cat([hidden_actor, hidden_map], dim=1)
            global_avails = torch.cat([actor_avails, polyline_avails], dim=1)

        # Perform self attention on all the features before sending it to the cross attention mask
        global_features = self.global_attention(global_features, mask=~global_avails)

        # Get the agent of interest to predict and add a new agent dimension of size 1 for compatibility 
        # with the decoder        
        batch_indices = torch.arange(history_features.shape[0], device=history_features.device)
        current_features = history_features[batch_indices, agent_to_predict, -1][:, None]
        current_availabilities = history_availabilities[batch_indices, agent_to_predict, -1][:, None]
        
        assert torch.all(current_availabilities), f"All current availabilities should be True "
            
        output = self.decoder(
            current_features,
            current_availabilities,
            global_features,
            global_avails,
        )
        return output
