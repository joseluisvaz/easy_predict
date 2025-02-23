from collections import defaultdict

import torch.nn.functional as F
from data_utils.feature_description import MAX_AGENTS_TO_PREDICT


import torch


import typing as T


def pad_second_dim(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Pads the second dimension of the tensor to the target size with zeros.
    Args:
        tensor (torch.Tensor): Input tensor of arbitrary dimensions.
        target_size (int): The size to pad the second dimension to.
    Returns:
        torch.Tensor: The padded tensor.
    """
    current_size = tensor.size(1)
    if current_size >= target_size:
        return tensor

    # Calculate the padding needed for the second dimension
    padding = (0, 0) * (tensor.dim() - 2) + (0, target_size - current_size)
    padded_tensor = F.pad(tensor, padding, "constant", 0)
    return padded_tensor


def batch_scenarios_by_feature(
    scenarios: T.Dict[str, T.List[T.Dict[str, torch.Tensor]]],
) -> T.Dict[str, torch.Tensor]:
    """Groups agent features by scenario and batches them together.
    Args:
        scenarios: Dictionary mapping scenario IDs to lists of agent feature dictionaries
    Returns:
        Dictionary mapping feature names to batched tensors
    """
    scenario_batch: T.List[T.Dict[str, torch.Tensor]] = []
    for list_of_agent_samples in scenarios.values():
        scenario: T.Dict[str, torch.Tensor] = {}
        for feature_key in list_of_agent_samples[0]:
            stacked_feature = torch.stack(
                [sample[feature_key] for sample in list_of_agent_samples], dim=0
            )[None, ...]
            scenario[feature_key] = pad_second_dim(
                stacked_feature, MAX_AGENTS_TO_PREDICT
            )
        scenario_batch.append(scenario)

    batched_scenarios = {}
    for feature_key in scenario_batch[0]:
        stacked_feature = torch.concatenate(
            [scenario[feature_key] for scenario in scenario_batch], dim=0
        )
        batched_scenarios[feature_key] = stacked_feature

    return batched_scenarios


def group_batch_by_scenario(
    batch: T.Dict[str, torch.Tensor], full_predicted_positions: torch.Tensor
) -> T.Dict[str, torch.Tensor]:
    """Groups batch samples by scenario ID and extracts relevant features for each agent.
    Args:
        batch: Dictionary containing batch data
        full_predicted_positions: Predicted trajectories for each agent
    Returns:
        Dictionary mapping scenario IDs to batched features
    """
    batch_size = batch["gt_features"].shape[0]

    # Group agent features by scenario
    scenarios: T.DefaultDict[str, T.List[T.Dict[str, torch.Tensor]]] = defaultdict(list)
    for i in range(batch_size):
        scenario_id = batch["scenario_id"][i].item()
        agent_id = batch["agent_to_predict"][i].item()
        agent_features = {
            "gt_states": batch["gt_states"][i][agent_id],
            "gt_states_avails": batch["gt_states_avails"][i][agent_id],
            "actor_type": batch["actor_type"][i][agent_id],
            "tracks_to_predict": batch["tracks_to_predict"][i][agent_id],
            "predicted_positions": full_predicted_positions[i][0],
        }
        scenarios[scenario_id].append(agent_features)

    return batch_scenarios_by_feature(scenarios)
