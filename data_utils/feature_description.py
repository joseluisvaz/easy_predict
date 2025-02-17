from collections import OrderedDict
from typing import Dict

import tensorflow as tf

SUBSAMPLE_SEQUENCE = 2  # e.g. 2 means 5 Hz instead of 10Hz sampling
NUM_HISTORY_FRAMES = 10 // SUBSAMPLE_SEQUENCE
NUM_FUTURE_FRAMES = 80 // SUBSAMPLE_SEQUENCE
NUM_MAP_SAMPLES = 30000

GT_STATES_MEANS = [16.9850, 2.4561, 4.4966, 2.0180, 1.6618, 0.5991, -0.0354]
GT_STATES_STDS = [41.7659, 29.3103, 1.2764, 0.3939, 3.4035, 5.1634, 2.5289]
ROADGRAPH_MEANS = [2.4929e01, 2.6036e00, 8.8117e-03, -1.2278e-03]
ROADGRAPH_STDS = [63.7964, 36.8537, 0.7378, 0.6223]

_ROADGRAPH_TYPE_TO_IDX = {
    "LaneCenter-Freeway": 1,
    "LaneCenter-SurfaceStreet": 2,
    "LaneCenter-BikeLane": 3,
    "RoadLine-BrokenSingleWhite": 6,
    "RoadLine-SolidSingleWhite": 7,
    "RoadLine-SolidDoubleWhite": 8,
    "RoadLine-BrokenSingleYellow": 9,
    "RoadLine-BrokenDoubleYellow": 10,
    "Roadline-SolidSingleYellow": 11,
    "Roadline-SolidDoubleYellow": 12,
    "RoadLine-PassingDoubleYellow": 13,
    "RoadEdgeBoundary": 15,
    "RoadEdgeMedian": 16,
    "StopSign": 17,
    "Crosswalk": 18,
    "SpeedBump": 19,
}

_ROADGRAPH_TYPE_TO_COLOR = {
    "LaneCenter-Freeway": "gray",
    "LaneCenter-SurfaceStreet": "gray",
    "LaneCenter-BikeLane": "gray",
    "RoadLine-BrokenSingleWhite": "white",
    "RoadLine-SolidSingleWhite": "white",
    "RoadLine-SolidDoubleWhite": "white",
    "RoadLine-BrokenSingleYellow": "yellow",
    "RoadLine-BrokenDoubleYellow": "yellow",
    "Roadline-SolidSingleYellow": "yellow",
    "Roadline-SolidDoubleYellow": "yellow",
    "RoadLine-PassingDoubleYellow": "yellow",
    "RoadEdgeBoundary": "purple",
    "RoadEdgeMedian": "purple",
    "StopSign": "red",
    "Crosswalk": "orange",
    "SpeedBump": "blue",
}


# Use this map to map the int elements of the roadgraph maps to a sequence of contiguous ints.
_ROADGRAPH_IDX_TO_TYPE = {idx: _type for _type, idx in _ROADGRAPH_TYPE_TO_IDX.items()}

_TL_STATUS_MAP = {
    "Unknown": 0,
    "Arrow_Stop": 1,
    "Arrow_Caution": 2,
    "Arrow_Go": 3,
    "Stop": 4,
    "Caution": 5,
    "Go": 6,
    "Flashing_Stop": 7,
    "Flashing_Caution": 8,
}

WAYMO_MAP_FEATURE_TO_COLOR = {
    0: "gray",
    1: "red",
    2: "yellow",
    3: "green",
    4: "red",
    5: "yellow",
    6: "green",
    7: "red",
    8: "yellow",
}

WAYMO_AGENT_TO_COLOR = {
    0: "gray",  # unset
    1: "teal",  # vehicle
    2: "pink",  # pedestrian
    3: "black",  # other
}

TRAFFIC_LIGHT_STATE_MAP = {
    0: "Unknown",
    1: "Arrow_Stop",
    2: "Arrow_Caution",
    3: "Arrow_Go",
    4: "Stop",
    5: "Caution",
    6: "Go",
    7: "Flashing_Stop",
    8: "Flashing_Caution",
}

ROADGRAPH_FEATURES = OrderedDict(
    [
        (
            "roadgraph_samples/dir",
            tf.io.FixedLenFeature([NUM_MAP_SAMPLES, 3], tf.float32, default_value=None),
        ),
        (
            "roadgraph_samples/id",
            tf.io.FixedLenFeature([NUM_MAP_SAMPLES, 1], tf.int64, default_value=None),
        ),
        (
            "roadgraph_samples/type",
            tf.io.FixedLenFeature([NUM_MAP_SAMPLES, 1], tf.int64, default_value=None),
        ),
        (
            "roadgraph_samples/valid",
            tf.io.FixedLenFeature([NUM_MAP_SAMPLES, 1], tf.int64, default_value=None),
        ),
        (
            "roadgraph_samples/xyz",
            tf.io.FixedLenFeature([NUM_MAP_SAMPLES, 3], tf.float32, default_value=None),
        ),
    ]
)

STATE_FEATURES = OrderedDict(
    [
        ("state/id", tf.io.FixedLenFeature([128], tf.float32, default_value=None)),
        ("state/type", tf.io.FixedLenFeature([128], tf.float32, default_value=None)),
        ("state/is_sdc", tf.io.FixedLenFeature([128], tf.int64, default_value=None)),
        ("state/tracks_to_predict", tf.io.FixedLenFeature([128], tf.int64, default_value=None)),
        ("state/objects_of_interest", tf.io.FixedLenFeature([128], tf.int64, default_value=None)),
        ("state/current/bbox_yaw", tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None)),
        ("state/current/height", tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None)),
        ("state/current/length", tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None)),
        (
            "state/current/timestamp_micros",
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        ),
        ("state/current/valid", tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None)),
        ("state/current/vel_yaw", tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None)),
        (
            "state/current/velocity_x",
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        ),
        (
            "state/current/velocity_y",
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        ),
        ("state/current/speed", tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None)),
        ("state/current/width", tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None)),
        ("state/current/x", tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None)),
        ("state/current/y", tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None)),
        ("state/current/z", tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None)),
        ("state/future/bbox_yaw", tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None)),
        ("state/future/height", tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None)),
        ("state/future/length", tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None)),
        (
            "state/future/timestamp_micros",
            tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
        ),
        ("state/future/valid", tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None)),
        ("state/future/vel_yaw", tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None)),
        (
            "state/future/velocity_x",
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        ),
        (
            "state/future/velocity_y",
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        ),
        (
            "state/future/speed",
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        ),
        (
            "state/future/width",
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        ),
        ("state/future/x", tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None)),
        ("state/future/y", tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None)),
        ("state/future/z", tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None)),
        ("state/past/bbox_yaw", tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None)),
        ("state/past/height", tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None)),
        ("state/past/length", tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None)),
        (
            "state/past/timestamp_micros",
            tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        ),
        ("state/past/valid", tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None)),
        ("state/past/vel_yaw", tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None)),
        ("state/past/velocity_x", tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None)),
        ("state/past/velocity_y", tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None)),
        ("state/past/speed", tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None)),
        ("state/past/width", tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None)),
        ("state/past/x", tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None)),
        ("state/past/y", tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None)),
        ("state/past/z", tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None)),
    ]
)

TRAFFIC_LIGHT_FEATURES = OrderedDict(
    [
        (
            "traffic_light_state/current/state",
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        ),
        (
            "traffic_light_state/current/valid",
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        ),
        (
            "traffic_light_state/current/x",
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        ),
        (
            "traffic_light_state/current/y",
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        ),
        (
            "traffic_light_state/current/z",
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        ),
        (
            "traffic_light_state/current/id",
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        ),
        (
            "traffic_light_state/past/state",
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        ),
        (
            "traffic_light_state/past/valid",
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        ),
        (
            "traffic_light_state/past/x",
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        ),
        (
            "traffic_light_state/past/y",
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        ),
        (
            "traffic_light_state/past/z",
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        ),
    ]
)


def get_feature_description() -> Dict:
    features_description = {}
    features_description.update(dict(ROADGRAPH_FEATURES))
    features_description.update(dict(STATE_FEATURES))
    features_description.update(TRAFFIC_LIGHT_FEATURES)
    return features_description
