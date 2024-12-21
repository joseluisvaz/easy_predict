import itertools
import io
from PIL import Image

import lightning as L
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from waymo_loader.dataloaders import WaymoH5Dataset, collate_waymo
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import numpy as np

from waymo_loader.feature_description import (
    _ROADGRAPH_IDX_TO_TYPE,
    _ROADGRAPH_TYPE_TO_COLOR,
    NUM_HISTORY_FRAMES,
    NUM_FUTURE_FRAMES,
)


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


class OnTrainCallback(L.Callback):
    def __init__(self, datadir: str, train_with_tracks_to_predict: bool):
        super().__init__()

        self._n_samples = 5
        dataset = WaymoH5Dataset(datadir, train_with_tracks_to_predict=train_with_tracks_to_predict)
        self._dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_waymo,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        for idx, single_sample_batch in enumerate(
            itertools.islice(self._dataloader, 10, 10 + self._n_samples)
        ):
            # Just extract a single sample from the batch and keep the batch dimension
            self.log_plot_to_clearml(
                pl_module,
                pl_module.task.get_logger(),
                single_sample_batch,
                trainer.current_epoch,
                idx,
            )

        train_metric_values = pl_module.metrics.result()
        for i, m in enumerate(["min_ade", "min_fde", "miss_rate", "overlap_rate", "map"]):
            for j, n in enumerate(pl_module.metrics.metric_names):
                self.log(f"metrics/{m}/{n}", float(train_metric_values[i, j]))
        pl_module.metrics.reset_state()

    def log_plot_to_clearml(self, pl_module, logger, single_sample, current_epoch, scene_idx: int):
        """Creates the image of a single scenario and logs it to ClearML.
        Assumption: Each component of the scenario has the batch dimension of size 1
        """

        # Make sure the input is in the gpu
        single_sample = {k: v.cuda() for k, v in single_sample.items()}

        with torch.no_grad():
            history_states = single_sample["gt_states"][:, :, : NUM_HISTORY_FRAMES + 1]
            history_avails = single_sample["gt_states_avails"][:, :, : NUM_HISTORY_FRAMES + 1]

            predicted_positions = pl_module.model(
                history_states,
                history_avails,
                single_sample["actor_type"],
                single_sample["roadgraph_features"],
                single_sample["roadgraph_features_mask"],
                single_sample["roadgraph_features_types"],
            )
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

        n_agents = history_states.shape[0]

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

            ground_truth_sequence = target_positions[
                agent_idx, target_availabilities[agent_idx], :2
            ]
            # Use the entire prediction horizon
            predicted_sequence = predicted_positions[agent_idx, :, :2]

            history_positions = history_sequence[..., :2]
            past_sequences.append(history_positions)
            ground_truth_sequences.append(ground_truth_sequence)
            predicted_sequences.append(predicted_sequence)
            current_states.append(current_state)

        map_features = torch.nested.to_padded_tensor(
            single_sample["roadgraph_features"], padding=0.0
        ).cpu()
        map_avails = torch.nested.to_padded_tensor(
            single_sample["roadgraph_features_mask"], padding=False
        ).cpu()
        map_types = torch.nested.to_padded_tensor(
            single_sample["roadgraph_features_types"], padding=False
        ).cpu()
        map_ids = torch.nested.to_padded_tensor(
            single_sample["roadgraph_features_ids"], padding=False
        ).cpu()

        map_features = map_features[..., :2].view(-1, 2)
        map_avails = map_avails.view(-1, 1)[:, 0]  # [N,]
        map_points = map_features[map_avails]  # [N, 2]
        map_types = map_types.view(-1)[map_avails].to(torch.int32)  # [N,]
        map_ids = map_ids.view(-1)[map_avails].to(torch.int32)  # [N,]

        plt.figure()

        unique_polyline_ids = torch.unique(map_ids)
        for polyline_id in unique_polyline_ids:
            filtered_points = map_points[map_ids == polyline_id]
            # type_id = map_types[map_ids == polyline_id][0]
            # color = _ROADGRAPH_TYPE_TO_COLOR[_ROADGRAPH_IDX_TO_TYPE[type_id.item()]]
            color = "gray"
            plt.plot(
                filtered_points[:, 0], filtered_points[:, 1], color=color, linewidth=0.5, zorder=0
            )

        for sequence in past_sequences:
            plt.scatter(sequence[:, 0], sequence[:, 1], color="gray", s=1.0, alpha=0.7, zorder=1)
        for sequence in ground_truth_sequences:
            plt.scatter(sequence[:, 0], sequence[:, 1], color="teal", s=2.0, alpha=0.7, zorder=1)

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
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)

        # Save the plot to an in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close()
        buf.seek(0)

        # Convert the buffer to a PIL Image
        image = Image.open(buf)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        logger.report_image(
            title="sample", series=f"scene{scene_idx}", iteration=current_epoch, image=image
        )
