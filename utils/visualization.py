import typing as T
from dataclasses import dataclass

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from data_utils.feature_description import (
    IDX_TO_ROADGRAPH_TYPE,
    MAX_AGENTS_TO_PREDICT,
    NUM_FUTURE_FRAMES,
    NUM_HISTORY_FRAMES,
    ROADGRAPH_TYPE_TO_COLOR,
)
from utils.geometry import get_transformation_matrix, transform_points

plt.style.use("dark_background")


def plot_oriented_box(
    ax: plt.Axes,
    x: float,
    y: float,
    orientation: float,
    length: float,
    width: float,
    color: str = "blue",
    alpha: float = 0.5,
    zorder: int = 1,
) -> None:
    """
    Plot an oriented box.
    Args:
        ax: matplotlib axis
        x, y: center coordinates
        orientation: rotation in radians
        length, width: box dimensions
        color: box color
        alpha: transparency
        zorder: plotting order
    """
    # Create a rectangle centered at origin
    rect = Rectangle(
        xy=(-length / 2, -width / 2),  # bottom-left corner
        width=length,
        height=width,
        color=color,
        alpha=alpha,
        zorder=zorder,
    )

    # Create transform: first rotate, then translate
    t = Affine2D().rotate(orientation).translate(x, y)
    rect.set_transform(t + ax.transData)

    # Add rectangle to plot
    ax.add_patch(rect)


def _plot_predicted_trajectories(
    predicted_positions: torch.Tensor,
    single_sample: T.Dict[str, torch.Tensor],
    anchor_index: int,
) -> None:
    """Plot predicted trajectories for each batch with a yellow-to-orange color gradient.

    Args:
        predicted_positions: Tensor of shape [batch_size, 1, num_timesteps, 2]
        single_sample: Dict containing scenario data
        anchor_index: Index of the anchor agent
    """
    predicted_positions = predicted_positions.cpu().numpy()
    predicted_sequences = []

    # Transform predictions to agent-centric coordinates for each batch
    for batch_idx in range(predicted_positions.shape[0]):
        # Get current state of anchor agent
        current_state = (
            single_sample["gt_states"][batch_idx, anchor_index, NUM_HISTORY_FRAMES, :]
            .cpu()
            .numpy()
        )
        current_yaw = current_state[4]

        # Transform predictions from world to agent coordinates
        world_to_agent = get_transformation_matrix(current_state[:2], current_yaw)
        predicted_sequence = predicted_positions[batch_idx, 0, :, :2]
        predicted_sequences.append(transform_points(predicted_sequence, world_to_agent))

    # Create color gradient from yellow to orange
    prediction_colors = LinearSegmentedColormap.from_list("", ["yellow", "orange"])(
        np.linspace(0, 1, predicted_sequences[0].shape[0])
    )

    # Plot each predicted trajectory
    for sequence in predicted_sequences:
        plt.scatter(
            sequence[:, 0],
            sequence[:, 1],
            color=prediction_colors,
            s=7.0,
            alpha=0.7,
            zorder=2,
        )


@dataclass
class AgentTrajectories:
    # Tensors of shape num_timesteps, 2
    history_sequences: T.List[np.ndarray]
    # Tensors of shape num_timesteps, 2
    future_sequences: T.List[np.ndarray]
    # Tensors of shape num_agents, 7
    current_states: T.List[np.ndarray]


def _process_agent_trajectories(
    single_sample: T.Dict[str, torch.Tensor], anchor_index: int
) -> AgentTrajectories:
    """Process the agent trajectories.

    Args:
        single_sample: A single sample from the validation set.
        anchor_index: The index of the anchor agent.

    Returns:
        AgentTrajectories: The processed agent trajectories.
    """
    # Crop the future positions to match the number of timesteps
    target_positions = (
        single_sample["gt_states"][anchor_index, :, -NUM_FUTURE_FRAMES:, :2]
        .cpu()
        .numpy()
    )
    target_availabilities = (
        single_sample["gt_states_avails"][anchor_index, :, -NUM_FUTURE_FRAMES:]
        .cpu()
        .numpy()
    )
    history_states = (
        single_sample["gt_states"][anchor_index, :, : NUM_HISTORY_FRAMES + 1]
        .cpu()
        .numpy()
    )
    history_availabilities = (
        single_sample["gt_states_avails"][anchor_index, :, : NUM_HISTORY_FRAMES + 1]
        .cpu()
        .numpy()
    )
    tracks_to_predict = single_sample["tracks_to_predict"][anchor_index].cpu().numpy()

    n_agents = history_states.shape[0]
    current_states = history_states[:, -1, :]
    current_availabilities = history_availabilities[:, -1]
    valid_current_states = [
        current_states[i] for i in range(n_agents) if current_availabilities[i]
    ]

    history_sequences = []
    ground_truth_sequences = []
    # The agents to predict are guaranteed to be in the first MAX_AGENTS_TO_PREDICT
    for agent_idx in range(MAX_AGENTS_TO_PREDICT):
        if tracks_to_predict[agent_idx] == 0:
            continue
        history_sequences.append(
            history_states[agent_idx, history_availabilities[agent_idx], :2]
        )
        ground_truth_sequences.append(
            target_positions[agent_idx, target_availabilities[agent_idx], :2]
        )

    return AgentTrajectories(
        history_sequences=history_sequences,
        future_sequences=ground_truth_sequences,
        current_states=valid_current_states,
    )


def _plot_agent_trajectories(
    single_sample: T.Dict[str, torch.Tensor], anchor_index: int
) -> None:
    """Plot the agent trajectories.

    Args:
        single_sample: A single sample from the validation set.
        anchor_index: The index of the anchor agent.
    """
    agent_trajectories = _process_agent_trajectories(single_sample, anchor_index)

    for sequence in agent_trajectories.history_sequences:
        plt.scatter(
            sequence[:, 0], sequence[:, 1], color="gray", s=1.0, alpha=0.7, zorder=1
        )
    for sequence in agent_trajectories.future_sequences:
        plt.scatter(
            sequence[:, 0], sequence[:, 1], color="teal", s=2.0, alpha=0.7, zorder=1
        )

    ax = plt.gca()
    for state in agent_trajectories.current_states:
        x, y, length, width, orientation, _, _ = state

        plot_oriented_box(
            ax,
            x,
            y,
            orientation,
            length,
            width,
            color="gray",
            zorder=3,
            alpha=0.9,
        )


@dataclass
class ProcessedMapFeatures:
    points: np.ndarray  # [P, N, 2]
    dirs: np.ndarray  # [P, N, 2]
    types: np.ndarray  # [P, N]
    avails: np.ndarray  # [P, N]


def _process_map_features(
    single_sample: T.Dict[str, torch.Tensor], anchor_index: int
) -> ProcessedMapFeatures:
    """Process the map features.

    Args:
        single_sample: A single sample from the validation set.
        anchor_index: The index of the anchor agent.

    Returns:
        ProcessedMapFeatures: The processed map features.
    """
    map_features = single_sample["roadgraph_features"][anchor_index].cpu().numpy()
    map_avails = single_sample["roadgraph_features_mask"][anchor_index].cpu().numpy()
    map_types = single_sample["roadgraph_features_types"][anchor_index].cpu().numpy()

    return ProcessedMapFeatures(
        points=map_features[..., :2],
        dirs=map_features[..., 2:4],
        types=map_types,
        avails=map_avails,
    )


def _plot_map_features(
    single_sample: T.Dict[str, torch.Tensor], anchor_index: int
) -> None:
    """Plot the map features.

    Args:
        single_sample: A single sample from the validation set.
        anchor_index: The index of the anchor agent.
    """
    processed_map_features = _process_map_features(single_sample, anchor_index)

    for polyline_id in range(processed_map_features.points.shape[0]):
        if not np.any(processed_map_features.avails[polyline_id]):
            continue

        avails = processed_map_features.avails[polyline_id]

        filtered_points = processed_map_features.points[polyline_id][avails]
        filtered_dirs = processed_map_features.dirs[polyline_id][avails]
        type_id = processed_map_features.types[polyline_id]

        if IDX_TO_ROADGRAPH_TYPE[type_id] in {"TYPE_CROSSWALK", "TYPE_SPEED_BUMP"}:
            filtered_points = np.concatenate([filtered_points, filtered_points[0:1]])

        color = ROADGRAPH_TYPE_TO_COLOR[IDX_TO_ROADGRAPH_TYPE[type_id.item()]]
        if IDX_TO_ROADGRAPH_TYPE[type_id] == "TYPE_STOP_SIGN":
            plt.scatter(
                filtered_points[:, 0],
                filtered_points[:, 1],
                color=color,
                linewidth=0.5,
                zorder=0,
                s=0.5,
            )
        else:
            plt.plot(
                filtered_points[:, 0],
                filtered_points[:, 1],
                color=color,
                linewidth=0.5,
                zorder=0,
            )
        if IDX_TO_ROADGRAPH_TYPE[type_id.item()] in {
            "TYPE_FREEWAY",
            "TYPE_SURFACE_STREET",
        }:
            plt.quiver(
                filtered_points[:, 0],
                filtered_points[:, 1],
                filtered_dirs[:, 0],
                filtered_dirs[:, 1],
                color=color,
                scale=1,
                scale_units="xy",
                angles="xy",
                width=0.002,
                headwidth=5,
                headlength=5,
                headaxislength=3,
                zorder=0,
            )


def plot_scene(
    single_sample: T.Dict[str, torch.Tensor],
    predicted_positions: T.Optional[torch.Tensor] = None,
    zoom_out: bool = False,
) -> None:
    """Plot a scene with the map features, agent trajectories, and predicted trajectories.

    Args:
        single_sample: A single sample from the validation set.
        predicted_positions: The predicted positions of the sample.
        zoom_out: Whether to zoom out the plot or to focus on the agent.
    """
    anchor_index = 0

    plt.figure()

    _plot_map_features(single_sample, anchor_index)
    _plot_agent_trajectories(single_sample, anchor_index)
    if predicted_positions is not None:
        _plot_predicted_trajectories(predicted_positions, single_sample, anchor_index)

    plt.title("predictions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    if not zoom_out:
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
