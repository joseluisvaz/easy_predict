from google.protobuf import text_format
import typing as T

from common_utils.agent_centric_to_scenario import (
    group_batch_by_scenario,
)
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

import tensorflow as tf
import torch
from data_utils.feature_description import MAX_AGENTS_TO_PREDICT


def _default_metrics_config():
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
        track_steps_per_second: 5
        prediction_steps_per_second: 5
        track_history_samples: 5
        track_future_samples: 40
        speed_lower_bound: 1.4
        speed_upper_bound: 11.0
        speed_scale_lower: 0.5
        speed_scale_upper: 1.0
        step_configurations {
          measurement_step: 5
          lateral_miss_threshold: 1.0
          longitudinal_miss_threshold: 2.0
        }
        step_configurations {
          measurement_step: 15
          lateral_miss_threshold: 1.8
          longitudinal_miss_threshold: 3.6
        }
        step_configurations {
          measurement_step: 39
          lateral_miss_threshold: 3.0
          longitudinal_miss_threshold: 6.0
        }
        max_predictions: 6
    """
    text_format.Parse(config_text, config)
    return config


class MotionMetrics(tf.keras.metrics.Metric):
    """Wrapper for motion metrics computation."""

    def __init__(self, config):
        super().__init__()
        self._prediction_trajectory = []
        self._prediction_score = []
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_ground_truth_indices = []
        self._prediction_ground_truth_indices_mask = []
        self._object_type = []
        self._metrics_config = config
        self.metric_names = config_util.get_breakdown_names_from_motion_config(config)

    def reset_state(self):
        self._prediction_trajectory = []
        self._prediction_score = []
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_ground_truth_indices = []
        self._prediction_ground_truth_indices_mask = []
        self._object_type = []

    def _update_state(
        self,
        prediction_trajectory: torch.Tensor,
        prediction_score: torch.Tensor,
        ground_truth_trajectory: torch.Tensor,
        ground_truth_is_valid: torch.Tensor,
        prediction_ground_truth_indices: torch.Tensor,
        prediction_ground_truth_indices_mask: torch.Tensor,
        object_type: torch.Tensor,
    ) -> None:
        def to_tf(torch_tensor: torch.Tensor) -> tf.Tensor:
            return tf.convert_to_tensor(torch_tensor.detach().cpu().numpy())

        self._prediction_trajectory.append(to_tf(prediction_trajectory))
        self._prediction_score.append(to_tf(prediction_score))
        self._ground_truth_trajectory.append(to_tf(ground_truth_trajectory))
        self._ground_truth_is_valid.append(to_tf(ground_truth_is_valid))
        self._prediction_ground_truth_indices.append(to_tf(prediction_ground_truth_indices))
        self._prediction_ground_truth_indices_mask.append(
            to_tf(prediction_ground_truth_indices_mask)
        )
        self._object_type.append(to_tf(object_type))

    def update_state(
        self, batch: T.Dict[str, torch.Tensor], full_predicted_positions: torch.Tensor
    ):
        batched_scenarios = group_batch_by_scenario(batch, full_predicted_positions)

        # Chop all the tensors to match the number of predicted agents the tracks to predict mask
        # will take care of just computing the metrics for the relevant agents
        predicted_positions = batched_scenarios["predicted_positions"][:, :MAX_AGENTS_TO_PREDICT]
        gt_states_avails = batched_scenarios["gt_states_avails"][:, :MAX_AGENTS_TO_PREDICT]
        gt_states = batched_scenarios["gt_states"][:, :MAX_AGENTS_TO_PREDICT]
        actor_type = batched_scenarios["actor_type"][:, :MAX_AGENTS_TO_PREDICT]
        tracks_to_predict_mask = batched_scenarios["tracks_to_predict"][:, :MAX_AGENTS_TO_PREDICT]

        batch_size, num_agents, _, _ = predicted_positions.shape
        # [batch_size, num_agents, steps, 2] -> # [batch_size, 1, 1, num_agents, steps, 2].
        # The added dimensions are top_k = 1, num_agents_per_joint_prediction = 1.
        predicted_positions = predicted_positions[:, None, None]
        # Fake the score since this model does not generate any score per predicted
        # trajectory. Get the first shapes [batch_size, num_preds, top_k] -> [batch_size, 1, 1].
        pred_score = torch.ones((batch_size, 1, 1))
        # [batch_size, num_pred, num_agents].
        pred_gt_indices = torch.arange(num_agents, dtype=torch.int64)
        pred_gt_indices = pred_gt_indices[None, None, :].expand(batch_size, 1, num_agents)
        # For the tracks to predict use the current timestamps
        pred_gt_indices_mask = tracks_to_predict_mask
        pred_gt_indices_mask = pred_gt_indices_mask.unsqueeze(1)

        self._update_state(
            prediction_trajectory=predicted_positions,
            prediction_score=pred_score,
            ground_truth_trajectory=gt_states,
            ground_truth_is_valid=gt_states_avails,
            prediction_ground_truth_indices=pred_gt_indices,
            prediction_ground_truth_indices_mask=pred_gt_indices_mask,
            object_type=actor_type,
        )

    def result(self):
        # [batch_size, num_preds, top_k, num_agents, steps, 2].
        prediction_trajectory = tf.concat(self._prediction_trajectory, 0)
        # [batch_size, num_preds, top_k].
        prediction_score = tf.concat(self._prediction_score, 0)
        # [batch_size, num_agents, gt_steps, 7].
        ground_truth_trajectory = tf.concat(self._ground_truth_trajectory, 0)
        # [batch_size, num_agents, gt_steps].
        ground_truth_is_valid = tf.concat(self._ground_truth_is_valid, 0)
        # [batch_size, num_preds, num_agents].
        prediction_ground_truth_indices = tf.concat(self._prediction_ground_truth_indices, 0)
        # [batch_size, num_preds, num_agents].
        prediction_ground_truth_indices_mask = tf.concat(
            self._prediction_ground_truth_indices_mask, 0
        )
        # [batch_size, num_agents].
        object_type = tf.cast(tf.concat(self._object_type, 0), tf.int64)

        # We are predicting more steps than needed by the eval code. Subsample.
        interval = (
            self._metrics_config.track_steps_per_second
            // self._metrics_config.prediction_steps_per_second
        )
        prediction_trajectory = prediction_trajectory[..., (interval - 1) :: interval, :]

        return py_metrics_ops.motion_metrics(
            config=self._metrics_config.SerializeToString(),
            prediction_trajectory=prediction_trajectory,
            prediction_score=prediction_score,
            ground_truth_trajectory=ground_truth_trajectory,
            ground_truth_is_valid=ground_truth_is_valid,
            prediction_ground_truth_indices=prediction_ground_truth_indices,
            prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
            object_type=object_type,
        )
