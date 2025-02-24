from typing import cast
import transforms3d

import numpy as np


def get_transformation_matrix(
    agent_translation_m: np.ndarray, agent_yaw: np.ndarray
) -> np.ndarray:
    """Get transformation matrix from world to vehicle frame"""

    # Translate world to ego by applying the negative ego translation.
    world_to_agent_in_2d = np.eye(3, dtype=np.float32)
    world_to_agent_in_2d[0:2, 2] = -agent_translation_m[0:2]

    # Rotate counter-clockwise by negative yaw to align world such that ego faces right.
    world_to_agent_in_2d = yaw_as_rotation33(-agent_yaw) @ world_to_agent_in_2d

    return world_to_agent_in_2d


def yaw_as_rotation33(yaw: float) -> np.ndarray:
    return transforms3d.euler.euler2mat(0, 0, yaw)


def rotation33_as_yaw(rotation: np.ndarray) -> float:
    return cast(float, transforms3d.euler.mat2euler(rotation)[2])


def transform_points(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """Transform points using transformation matrix.
    Note this function assumes points.shape[1] == matrix.shape[1] - 1, which means that the last
    row in the matrix does not influence the final result.
    For 2D points only the first 2x3 part of the matrix will be used.

    Args:
        points (np.ndarray): Input points (Nx2) or (Nx3).
        transf_matrix (np.ndarray): 3x3 or 4x4 transformation matrix for 2D and 3D input respectively

    Returns:
        np.ndarray: array of shape (N,2) for 2D input points, or (N,3) points for 3D input points
    """
    assert len(points.shape) == len(transf_matrix.shape) == 2, (
        f"dimensions mismatch, both points ({points.shape}) and "
        f"transf_matrix ({transf_matrix.shape}) needs to be 2D numpy ndarrays."
    )
    assert transf_matrix.shape[0] == transf_matrix.shape[1], (
        f"transf_matrix ({transf_matrix.shape}) should be a square matrix."
    )

    if points.shape[1] not in [2, 3]:
        raise AssertionError(
            f"Points input should be (N, 2) or (N, 3) shape, received {points.shape}"
        )

    assert points.shape[1] == transf_matrix.shape[1] - 1, (
        "points dim should be one less than matrix dim"
    )

    return (points @ transf_matrix.T[:-1, :-1]) + transf_matrix[:-1, -1]


def get_so2_from_se2(transform_se3: np.ndarray) -> np.ndarray:
    """Gets rotation component in SO(2) from transformation in SE(2).

    Args:
        transform_se3: se2 transformation.

    Returns:
        rotation component in so2
    """
    rotation = np.eye(3)
    rotation[:2, :2] = transform_se3[:2, :2]
    return rotation


def get_yaw_from_se2(transform_se3: np.ndarray) -> float:
    """Gets yaw from transformation in SE(2).

    Args:
        transform_se3: se2 transformation.

    Returns:
        yaw
    """
    return rotation33_as_yaw(get_so2_from_se2(transform_se3))
