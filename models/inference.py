import typing as T

import torch

from data_utils.feature_description import NUM_HISTORY_FRAMES
from models.prediction import PredictionModel


def run_model_forward_pass(
    model: PredictionModel, batch: T.Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Run model inference normalizing the feature vectors and the output of the model
    Args:
        model (PredictionModel): _description_
        batch (T.Dict[str, torch.Tensor]): _description_
        normalized (bool, optional): _description_. Defaults to False.

    Returns:
        torch.Tensor: _description_
    """
    history_states = batch["gt_features"][:, :, : NUM_HISTORY_FRAMES + 1, :]
    history_avails = batch["gt_features_avails"][:, :, : NUM_HISTORY_FRAMES + 1]

    predicted_positions = model(
        history_states,
        history_avails,
        batch["actor_type"],
        batch["roadgraph_features"],
        batch["roadgraph_features_mask"],
        batch["roadgraph_features_types"],
        batch["tl_states"],
        batch["tl_states_categorical"],
        batch["tl_avails"],
        batch["tracks_to_predict"],
        batch["agent_to_predict"],
    )

    return predicted_positions
