from unittest.mock import MagicMock

import numpy as np
import pytest

from data_utils.proto_decoding import _create_agent_masks


@pytest.fixture
def mock_scenario() -> MagicMock:
    # Create a mock scenario object
    scenario = MagicMock()
    scenario.tracks = [MagicMock() for _ in range(5)]  # Assume 5 agents
    scenario.tracks_to_predict = [
        MagicMock(track_index=1),
        MagicMock(track_index=3),
    ]
    scenario.sdc_track_index = 2
    return scenario


def test_create_agent_masks(mock_scenario: MagicMock) -> None:
    tracks_to_predict, is_sdc = _create_agent_masks(mock_scenario)

    # Check the shape and type of the output
    assert tracks_to_predict.shape == (5,)
    assert is_sdc.shape == (5,)
    assert tracks_to_predict.dtype == np.bool_
    assert is_sdc.dtype == np.bool_

    # Check the values in the output arrays
    expected_tracks_to_predict = np.array([False, True, False, True, False])
    expected_is_sdc = np.array([False, False, True, False, False])

    np.testing.assert_array_equal(tracks_to_predict, expected_tracks_to_predict)
    np.testing.assert_array_equal(is_sdc, expected_is_sdc)
