from google.protobuf import text_format

from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

import tensorflow as tf
from torch import Tensor as TorchTensor
from waymo_loader.feature_description import NUM_FUTURE_FRAMES, NUM_HISTORY_FRAMES, SUBSAMPLE_SEQUENCE 


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

    def update_state(
        self,
        prediction_trajectory: TorchTensor,
        prediction_score: TorchTensor,
        ground_truth_trajectory: TorchTensor,
        ground_truth_is_valid: TorchTensor,
        prediction_ground_truth_indices: TorchTensor,
        prediction_ground_truth_indices_mask: TorchTensor,
        object_type: TorchTensor,
    ):
        def to_tf(torch_tensor):
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
