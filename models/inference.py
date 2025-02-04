import typing as T

import torch

from models.prediction import PredictionModel
from waymo_loader.feature_description import NUM_HISTORY_FRAMES


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
    history_states = batch["gt_states"][:, :, : NUM_HISTORY_FRAMES + 1, :]
    history_avails = batch["gt_states_avails"][:, :, : NUM_HISTORY_FRAMES + 1]
    roadgraph_feats = batch["roadgraph_features"]
    roadgraph_avails = batch["roadgraph_features_mask"]

    predicted_positions = model(
        history_states,
        history_avails,
        batch["actor_type"],
        roadgraph_feats,
        roadgraph_avails,
        batch["roadgraph_features_types"],
    )

    return predicted_positions
