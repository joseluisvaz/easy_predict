import typing as T

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from data_utils.feature_description import NUM_FUTURE_FRAMES, NUM_HISTORY_FRAMES, _ROADGRAPH_TYPE_TO_COLOR, _ROADGRAPH_IDX_TO_TYPE

plt.style.use("dark_background")

def plot_oriented_box(ax, x, y, orientation, length, width, color="blue", alpha=0.5, zorder=1):
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


def plot_scene(
    single_sample,
    predicted_positions: T.Optional[torch.Tensor] = None,
    zoom_out: bool = False,
    plot_gray: bool = True,
) -> None:

    if predicted_positions is not None:
        predicted_positions = predicted_positions[0].cpu().numpy()

    # Crop the future positions to match the number of timesteps
    target_positions = single_sample["gt_states"][0, :, -NUM_FUTURE_FRAMES:, :2].cpu().numpy()
    target_availabilities = (
        single_sample["gt_states_avails"][0, :, -NUM_FUTURE_FRAMES:].cpu().numpy()
    )
    history_states = single_sample["gt_states"][0, :, : NUM_HISTORY_FRAMES + 1].cpu().numpy()
    history_availabilities = (
        single_sample["gt_states_avails"][0, :, : NUM_HISTORY_FRAMES + 1].cpu().numpy()
    )

    # n_agents = history_states.shape[0]
    n_agents = 8

    past_sequences = []
    ground_truth_sequences = []
    predicted_sequences = []
    current_states = []
    for agent_idx in range(n_agents):
        if history_availabilities[agent_idx, -1] == 0:
            # Agent is not available in the current timestep
            continue

        history_sequence = history_states[agent_idx, history_availabilities[agent_idx], :]
        current_state = history_sequence[-1]

        ground_truth_sequence = target_positions[agent_idx, target_availabilities[agent_idx], :2]

        history_positions = history_sequence[..., :2]
        past_sequences.append(history_positions)
        ground_truth_sequences.append(ground_truth_sequence)
        current_states.append(current_state)

        if predicted_positions is not None:
            predicted_sequence = predicted_positions[agent_idx, :, :2]
            predicted_sequences.append(predicted_sequence)

    map_features = single_sample["roadgraph_features"].cpu()
    map_avails = single_sample["roadgraph_features_mask"].cpu()
    map_types = single_sample["roadgraph_features_types"].cpu()
    map_ids = single_sample["roadgraph_features_ids"].cpu()

    n_batch, n_polylines, n_points, _ = map_features.shape
    map_features = map_features[..., :4].view(-1, 4)

    map_avails = map_avails.view(-1, 1)[:, 0]  # [N,]
    map_points = map_features[..., :2][map_avails]  # [N, 2]
    map_dirs = map_features[..., 2:4][map_avails]  # [N, 2]

    map_types = map_types.view(n_batch, n_polylines, 1).expand(-1, -1, n_points)
    map_ids = map_ids.view(n_batch, n_polylines, 1).expand(-1, -1, n_points)

    map_types = map_types.flatten()[map_avails].to(torch.int32)  # [N,]
    map_ids = map_ids.flatten()[map_avails].to(torch.int32)  # [N,]

    plt.figure()

    unique_polyline_ids = torch.unique(map_ids)
    for polyline_id in unique_polyline_ids:
        filtered_points = map_points[map_ids == polyline_id]
        filtered_dirs = map_dirs[map_ids == polyline_id]
        type_id = map_types[map_ids == polyline_id][0]
        color = _ROADGRAPH_TYPE_TO_COLOR[_ROADGRAPH_IDX_TO_TYPE[type_id.item()]]
        # color = "gray" if plot_gray else None
        if _ROADGRAPH_IDX_TO_TYPE[type_id.item()] == "StopSign":
            plt.scatter(filtered_points[:, 0], filtered_points[:, 1], color=color, linewidth=0.5, zorder=0, s=0.5)
        else:
            plt.plot(filtered_points[:, 0], filtered_points[:, 1], color=color, linewidth=0.5, zorder=0)
        if "LaneCenter" in _ROADGRAPH_IDX_TO_TYPE[type_id.item()]:
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

    for sequence in past_sequences:
        plt.scatter(sequence[:, 0], sequence[:, 1], color="gray", s=1.0, alpha=0.7, zorder=1)
    for sequence in ground_truth_sequences:
        plt.scatter(sequence[:, 0], sequence[:, 1], color="teal", s=2.0, alpha=0.7, zorder=1)

    if predicted_positions is not None:
        prediction_colors = LinearSegmentedColormap.from_list("", ["yellow", "orange"])(
            np.linspace(0, 1, predicted_sequences[0].shape[0])
        )
        for sequence in predicted_sequences:
            plt.scatter(
                sequence[:, 0], sequence[:, 1], color=prediction_colors, s=7.0, alpha=0.7, zorder=2
            )

    ax = plt.gca()
    for state in current_states:
        x, y, length, width, orientation, _, _ = state

        # orientation = np.arctan2(s, c)
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

    plt.title("predictions")
    plt.xlabel("x")
    plt.ylabel("y")
    if not zoom_out:
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
