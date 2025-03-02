from collections import defaultdict
from typing import Any, Optional

import numpy as np
from waymo_open_dataset.protos import scenario_pb2

from data_utils.feature_description import (
    MAX_AGENTS_IN_SCENARIO,
    ROADGRAPH_TYPE_TO_IDX,
    SEQUENCE_LENGTH,
    SUBSAMPLE_SEQUENCE,
)

POLYLINE_SUBSAMPLE_FACTOR = 4
MAX_POLYLINE_LENGTH = 20
MAX_NUM_POLYLINES = 800
MAX_NUM_TL = 16
MAX_NUM_TL_TIMES = 11
MAX_ORIGINAL_SEQUENCE_LENGTH = 91  # (10Hz, 9.1s)

LANE_TYPE_TO_GLOBAL_TYPE = {
    0: "TYPE_UNDEFINED",
    1: "TYPE_FREEWAY",
    2: "TYPE_SURFACE_STREET",
    3: "TYPE_BIKE_LANE",
}

ROAD_LINE_TYPE_TO_GLOBAL_TYPE = {
    0: "TYPE_UNKNOWN",
    1: "TYPE_BROKEN_SINGLE_WHITE",
    2: "TYPE_SOLID_SINGLE_WHITE",
    3: "TYPE_SOLID_DOUBLE_WHITE",
    4: "TYPE_BROKEN_SINGLE_YELLOW",
    5: "TYPE_BROKEN_DOUBLE_YELLOW",
    6: "TYPE_SOLID_SINGLE_YELLOW",
    7: "TYPE_SOLID_DOUBLE_YELLOW",
    8: "TYPE_PASSING_DOUBLE_YELLOW",
}

ROAD_EDGE_TYPE_TO_GLOBAL_TYPE = {
    0: "TYPE_UNKNOWN",
    1: "TYPE_ROAD_EDGE_BOUNDARY",
    2: "TYPE_ROAD_EDGE_MEDIAN",
}


TL_SIGNAL_IDX_TO_STATE = {
    0: "LANE_STATE_UNKNOWN",
    # // States for traffic signals with arrows.
    1: "LANE_STATE_ARROW_STOP",
    2: "LANE_STATE_ARROW_CAUTION",
    3: "LANE_STATE_ARROW_GO",
    # // Standard round traffic signals.
    4: "LANE_STATE_STOP",
    5: "LANE_STATE_CAUTION",
    6: "LANE_STATE_GO",
    # // Flashing light signals.
    7: "LANE_STATE_FLASHING_STOP",
    8: "LANE_STATE_FLASHING_CAUTION",
}


def _create_agent_masks(
    scenario: scenario_pb2.Scenario,
) -> tuple[np.ndarray, np.ndarray]:
    """Create masks for the agents in the scenario."""
    num_agents_in_scenario = len(scenario.tracks)

    tracks_to_predict = np.zeros(num_agents_in_scenario, dtype=np.bool_)
    for required_prediction in scenario.tracks_to_predict:
        tracks_to_predict[required_prediction.track_index] = True

    is_sdc = np.zeros(num_agents_in_scenario, dtype=np.bool_)
    is_sdc[scenario.sdc_track_index] = True

    return tracks_to_predict, is_sdc


def _extract_agent_states(
    agent: scenario_pb2.Track,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract states and availability from a single agent."""
    states = [
        np.array(
            [
                x.center_x,
                x.center_y,
                x.length,
                x.width,
                x.heading,
                x.velocity_x,
                x.velocity_y,
            ],
            dtype=np.float32,
        )
        for x in agent.states
    ]
    assert len(states) == MAX_ORIGINAL_SEQUENCE_LENGTH, (
        f"sequence length is {len(states)}"
    )
    avails = [np.array(x.valid, dtype=np.bool_) for x in agent.states]

    return (
        states[::SUBSAMPLE_SEQUENCE],
        avails[::SUBSAMPLE_SEQUENCE],
        agent.object_type,
    )


def pad_or_truncate(features: np.ndarray, max_size: int, padding: Any) -> np.ndarray:
    """Pad or truncate the features to the max size in the first dimension."""
    if features.shape[0] >= max_size:
        return features[:max_size]

    padding_shape = list(features.shape[1:])
    padding_shape.insert(0, max_size - features.shape[0])
    return np.concatenate(
        (
            features,
            np.full(padding_shape, padding, dtype=features.dtype),
        ),
        axis=0,
    )


def decode_agent_features(scenario: scenario_pb2.Scenario) -> dict[str, np.ndarray]:
    tracks_to_predict, is_sdc = _create_agent_masks(scenario)

    features_as_lists = defaultdict(list)
    for agent in scenario.tracks:
        states, avails, actor_type = _extract_agent_states(agent)
        features_as_lists["gt_states"].append(states)
        features_as_lists["gt_states_avails"].append(avails)
        features_as_lists["actor_type"].append(actor_type)

    features = {}
    # Stack features in the agent dimension
    for key in ["gt_states", "gt_states_avails", "actor_type"]:
        features[key] = np.stack(features_as_lists[key])

    features["tracks_to_predict"] = tracks_to_predict.copy()
    features["is_sdc"] = is_sdc.copy()

    def _order_predictable_agents(feature: np.ndarray, padding: Any) -> np.ndarray:
        all_agents = np.concatenate(
            [
                feature[tracks_to_predict | is_sdc],
                feature[~tracks_to_predict & ~is_sdc],
            ],
            axis=0,
        )
        assert len(all_agents) == len(feature), "needs to be the same length"
        return pad_or_truncate(all_agents, MAX_AGENTS_IN_SCENARIO, padding)

    # Order and pad all features
    for key in features.keys():
        features[key] = _order_predictable_agents(features[key], padding=0)

    # Make sure output types are correct, TODO: move to unit tests
    output_types_and_expected_sizes = {
        "gt_states": (np.float32, (MAX_AGENTS_IN_SCENARIO, SEQUENCE_LENGTH, 7)),
        "gt_states_avails": (np.bool_, (MAX_AGENTS_IN_SCENARIO, SEQUENCE_LENGTH)),
        "actor_type": (np.int64, (MAX_AGENTS_IN_SCENARIO,)),
        "is_sdc": (np.bool_, (MAX_AGENTS_IN_SCENARIO,)),
        "tracks_to_predict": (np.bool_, (MAX_AGENTS_IN_SCENARIO,)),
    }

    for key in output_types_and_expected_sizes.keys():
        assert features[key].shape == output_types_and_expected_sizes[key][1], (
            f"{key} has shape {features[key].shape} but should have shape {output_types_and_expected_sizes[key][1]}"
        )
        assert features[key].dtype == output_types_and_expected_sizes[key][0], (
            f"{key} has dtype {features[key].dtype} but should have dtype {output_types_and_expected_sizes[key][0]}"
        )

    return features


def get_polyline_dir(polyline: np.ndarray) -> np.ndarray:
    "adapted from: https://github.com/sshaoshuai/MTR/blob/master/mtr/datasets/waymo/data_preprocess.py"
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(
        np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000
    )
    return polyline_dir


def _get_polyline_features(points: np.ndarray, global_type: int) -> np.ndarray:
    """Get the polyline features for a given points and global type."""
    cur_polyline = np.stack(
        [np.array([point.x, point.y, global_type]) for point in points],
        axis=0,
    )
    cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:2])
    return np.concatenate(
        (cur_polyline[:, 0:2], cur_polyline_dir, cur_polyline[:, 2:]), axis=-1
    )


def _subsample_polyline(polyline: np.ndarray, subsample: int) -> np.ndarray:
    if len(polyline) <= 3:
        return polyline
    indices = np.arange(1, len(polyline) - 1, subsample)
    indices = np.concatenate(([0], indices, [len(polyline) - 1]))
    return polyline[indices]


def _get_type_from_map_feature(map_feature: Any) -> Optional[int]:
    if map_feature.lane.ByteSize() > 0:
        if map_feature.lane.type == 0:
            return None
        return ROADGRAPH_TYPE_TO_IDX[LANE_TYPE_TO_GLOBAL_TYPE[map_feature.lane.type]]
    elif map_feature.road_line.ByteSize() > 0:
        if map_feature.road_line.type == 0:
            return None
        return ROADGRAPH_TYPE_TO_IDX[
            ROAD_LINE_TYPE_TO_GLOBAL_TYPE[map_feature.road_line.type]
        ]
    elif map_feature.road_edge.ByteSize() > 0:
        if map_feature.road_edge.type == 0:
            return None
        return ROADGRAPH_TYPE_TO_IDX[
            ROAD_EDGE_TYPE_TO_GLOBAL_TYPE[map_feature.road_edge.type]
        ]
    elif map_feature.stop_sign.ByteSize() > 0:
        return ROADGRAPH_TYPE_TO_IDX["TYPE_STOP_SIGN"]
    elif map_feature.crosswalk.ByteSize() > 0:
        return ROADGRAPH_TYPE_TO_IDX["TYPE_CROSSWALK"]
    elif map_feature.speed_bump.ByteSize() > 0:
        return ROADGRAPH_TYPE_TO_IDX["TYPE_SPEED_BUMP"]
    else:
        raise ValueError


def _get_polyline_from_map_feature(
    map_feature: Any, global_type: int
) -> Optional[np.ndarray]:
    if map_feature.lane.ByteSize() > 0:
        return _get_polyline_features(map_feature.lane.polyline, global_type)
    elif map_feature.road_line.ByteSize() > 0:
        return _get_polyline_features(map_feature.road_line.polyline, global_type)
    elif map_feature.road_edge.ByteSize() > 0:
        return _get_polyline_features(map_feature.road_edge.polyline, global_type)
    elif map_feature.stop_sign.ByteSize() > 0:
        point = map_feature.stop_sign.position
        return np.array([point.x, point.y, 0, 0, global_type]).reshape(1, 5)
    elif map_feature.crosswalk.ByteSize() > 0:
        return _get_polyline_features(map_feature.crosswalk.polygon, global_type)
    elif map_feature.speed_bump.ByteSize() > 0:
        return _get_polyline_features(map_feature.speed_bump.polygon, global_type)
    else:
        raise ValueError


def _subsample_polyline_by_type(polyline: np.ndarray, global_type: int) -> np.ndarray:
    """Subsample the polyline by type."""
    # TODO: make the type and int, currently it is a float on the same tensor
    if global_type == ROADGRAPH_TYPE_TO_IDX["TYPE_STOP_SIGN"]:
        return polyline
    else:
        return _subsample_polyline(polyline, POLYLINE_SUBSAMPLE_FACTOR)


def _decode_map_features_into_polylines(
    scenario: scenario_pb2.Scenario,
) -> list[np.ndarray]:
    "adapted from: https://github.com/sshaoshuai/MTR/blob/master/mtr/datasets/waymo/data_preprocess.py"

    polylines = []
    for cur_data in scenario.map_features:
        if cur_data.driveway.ByteSize() > 0:
            # We are ignoring driveways for now
            continue

        global_type = _get_type_from_map_feature(cur_data)
        if global_type is None:
            continue

        cur_polyline = _get_polyline_from_map_feature(cur_data, global_type)
        cur_polyline = _subsample_polyline_by_type(cur_polyline, global_type)

        decomposed_polyline = []
        for i in range(0, len(cur_polyline), MAX_POLYLINE_LENGTH):
            decomposed_polyline.append(cur_polyline[i : i + MAX_POLYLINE_LENGTH])

        for polyline in decomposed_polyline:
            assert len(polyline) <= MAX_POLYLINE_LENGTH
            polylines.append(polyline)
            if len(polylines) >= MAX_NUM_POLYLINES:
                break

        if len(polylines) >= MAX_NUM_POLYLINES:
            break

    assert len(polylines) <= MAX_NUM_POLYLINES
    return polylines


def _decode_traffic_light_features_into_lists(
    scenario: scenario_pb2.Scenario,
) -> list[np.ndarray]:
    dynamic_map_features = scenario.dynamic_map_states
    current_time_index = scenario.current_time_index

    features_per_timestamp = []

    maximum_sequence_size = min(current_time_index + 1, MAX_NUM_TL_TIMES)
    for map_feature in dynamic_map_features[:maximum_sequence_size]:  # num timestamps
        if len(map_feature.lane_states) == 0:
            continue

        traffic_light_features = []
        for cur_signal in map_feature.lane_states[:MAX_NUM_TL]:
            traffic_light_features.append(
                [
                    cur_signal.stop_point.x,
                    cur_signal.stop_point.y,
                    cur_signal.state,
                ]
            )

        features_per_timestamp.append(
            np.array(traffic_light_features).astype(np.float32)
        )

    assert len(features_per_timestamp) <= MAX_NUM_TL_TIMES, (
        "number of traffic lights is too long"
    )
    return features_per_timestamp


def decode_map_features(scenario: scenario_pb2.Scenario) -> dict[str, np.ndarray]:
    polyline_tensor = np.zeros(
        (MAX_NUM_POLYLINES, MAX_POLYLINE_LENGTH, 5), dtype=np.float32
    )
    polyline_type_tensor = np.zeros((MAX_NUM_POLYLINES,), dtype=np.int64)
    polyline_mask_tensor = np.zeros(
        (MAX_NUM_POLYLINES, MAX_POLYLINE_LENGTH), dtype=np.bool_
    )

    polylines = _decode_map_features_into_polylines(scenario)

    for i, polyline in enumerate(polylines):
        polyline_tensor[i, : len(polyline)] = polyline
        polyline_mask_tensor[i, : len(polyline)] = 1
        polyline_type_tensor[i] = polyline[-1, -1]

    return {
        "roadgraph_features": polyline_tensor,
        "roadgraph_features_mask": polyline_mask_tensor,
        "roadgraph_features_types": polyline_type_tensor,
    }


def decode_traffic_light_features(
    scenario: scenario_pb2.Scenario,
) -> dict[str, np.ndarray]:
    tl_states = np.zeros((MAX_NUM_TL_TIMES, MAX_NUM_TL, 2), dtype=np.float32)
    tl_states_categorical = np.zeros((MAX_NUM_TL_TIMES, MAX_NUM_TL), dtype=np.int64)
    tl_states_avails = np.zeros((MAX_NUM_TL_TIMES, MAX_NUM_TL), dtype=np.bool_)

    features_per_timestamp = _decode_traffic_light_features_into_lists(scenario)

    for i, traffic_lights in enumerate(features_per_timestamp):
        tl_states[i, : len(traffic_lights), :2] = traffic_lights[:, :2]
        tl_states_categorical[i, : len(traffic_lights)] = traffic_lights[:, 2]
        tl_states_avails[i, : len(traffic_lights)] = 1

    return {
        "tl_states": tl_states.transpose(1, 0, 2),
        "tl_states_categorical": tl_states_categorical.transpose(1, 0),
        "tl_avails": tl_states_avails.transpose(1, 0),
    }


def generate_features_from_proto(
    scenario: scenario_pb2.Scenario,
) -> dict[str, np.ndarray]:
    features_dict: dict[str, np.ndarray] = {}
    features_dict.update(decode_agent_features(scenario))
    features_dict.update(decode_map_features(scenario))
    features_dict.update(decode_traffic_light_features(scenario))

    # gt_states: (x, y, length, width, yaw, velocity_x, velocity_y)
    # roadgraph_features: (x, y, z, dx, dy, dz, type)
    # tl_states: (x, y, z)
    feature_to_shape_and_type = {
        "gt_states": (np.float32, (MAX_AGENTS_IN_SCENARIO, SEQUENCE_LENGTH, 7)),
        "gt_states_avails": (np.bool_, (MAX_AGENTS_IN_SCENARIO, SEQUENCE_LENGTH)),
        "actor_type": (np.int64, (MAX_AGENTS_IN_SCENARIO,)),
        "is_sdc": (np.bool_, (MAX_AGENTS_IN_SCENARIO,)),
        "tracks_to_predict": (np.bool_, (MAX_AGENTS_IN_SCENARIO,)),
        "roadgraph_features": (np.float32, (MAX_NUM_POLYLINES, MAX_POLYLINE_LENGTH, 5)),
        "roadgraph_features_mask": (np.bool_, (MAX_NUM_POLYLINES, MAX_POLYLINE_LENGTH)),
        "roadgraph_features_types": (np.int64, (MAX_NUM_POLYLINES,)),
        "tl_states": (np.float32, (MAX_NUM_TL, MAX_NUM_TL_TIMES, 2)),
        "tl_states_categorical": (np.int64, (MAX_NUM_TL, MAX_NUM_TL_TIMES)),
        "tl_avails": (np.bool_, (MAX_NUM_TL, MAX_NUM_TL_TIMES)),
    }

    for key in feature_to_shape_and_type.keys():
        assert features_dict[key].dtype == feature_to_shape_and_type[key][0], (
            f"{key} has dtype {features_dict[key].dtype} but should have dtype {feature_to_shape_and_type[key][0]}"
        )
        assert features_dict[key].shape == feature_to_shape_and_type[key][1], (
            f"{key} has shape {features_dict[key].shape} but should have shape {feature_to_shape_and_type[key][1]}"
        )

    return features_dict
