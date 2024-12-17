from typing import TypeVar


import torch
import numpy as np
from torch import Tensor

Array = TypeVar("Array", np.ndarray, Tensor)

def force_pad_batch_size(tensor: Array, max_batch_size: int) -> Array:
    """ Pads or crops a tensor to an specific max_batch_size
    When padding, this function will do exclusively zero padding
    Args:
        tensor: tensor to pad or crop
        max_batch_size: max batch size of tensor
    Returns:
        the cropped or padded tensor with max_batch_size
    """

    def force_pad_batch_size_np(tensor: np.ndarray, max_batch_size: int) -> np.ndarray:
        # If tensor is too big then just crop it to the right size
        if max_batch_size - tensor.shape[0] < 0:
            return tensor[:max_batch_size]

        # Create padding for all dimensions but only set to non zero for the batch dimension
        padding = tensor.ndim * [
            [0, 0],
        ]
        padding[0] = [0, (max_batch_size - tensor.shape[0])]
        padded_mat = np.pad(tensor, padding, "constant", constant_values=(0, 0))
        return padded_mat

    if isinstance(tensor, np.ndarray):
        return force_pad_batch_size_np(tensor, max_batch_size)
    elif isinstance(tensor, Tensor):
        return torch.from_numpy(force_pad_batch_size_np(tensor.numpy(), max_batch_size))
    else:
        raise TypeError("valid tensor types: torch.tensor | np.ndarray")

def flatten_agent_dim(_input: Tensor) -> Tensor:
    return _input.reshape(_input.size(0) * _input.size(1), *_input.shape[2:])

def expand_agent_dim(_input: Tensor, n_batch: int, n_agents: int) -> Tensor:
    assert n_batch * n_agents == _input.size(0)
    return _input.view(n_batch, n_agents, *_input.shape[1:])
