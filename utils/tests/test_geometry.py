import numpy as np

from utils.geometry import transform_points


def test_transform_points_2d() -> None:
    # Test 2D points with translation
    points = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    translation = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])

    result = transform_points(points, translation)
    expected = np.array([[3.0, 3.0], [2.0, 4.0], [3.0, 4.0]])
    np.testing.assert_array_almost_equal(result, expected)

    # Test 2D points with rotation (90 degrees counterclockwise)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    result = transform_points(points, rotation)
    expected = np.array([[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]])
    np.testing.assert_array_almost_equal(result, expected)


def test_transform_points_3d() -> None:
    # Test 3D points with translation
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    result = transform_points(points, translation)
    expected = np.array([[3.0, 3.0, 4.0], [2.0, 4.0, 4.0], [2.0, 3.0, 5.0]])
    np.testing.assert_array_almost_equal(result, expected)
