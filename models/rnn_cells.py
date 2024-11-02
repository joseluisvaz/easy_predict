from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from common_utils.tensor_utils import flatten_agent_dim, expand_agent_dim


class MultiAgentLSTMCell(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.inner_lstm_cell = nn.LSTMCell(in_dim, out_dim)

    def get_initial_hidden_state(
        self, shape: tuple, device: torch.device, requires_grad: bool = False
    ) -> Tuple[Tensor, Tensor]:
        h = torch.zeros(shape, requires_grad=requires_grad, device=device)
        c = torch.zeros(shape, requires_grad=requires_grad, device=device)
        return h, c

    def forward(
        self, _input: Tensor, hidden: Tuple[Tensor, Tensor], mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        n_batch, n_agents, _ = _input.shape

        h_flat = flatten_agent_dim(hidden[0])
        c_flat = flatten_agent_dim(hidden[1])

        _input = flatten_agent_dim(_input)

        h_next, c_next = self.inner_lstm_cell(_input, (h_flat, c_flat))

        h_next = expand_agent_dim(h_next, n_batch, n_agents)
        c_next = expand_agent_dim(c_next, n_batch, n_agents)

        # Mask unvalid hidden states
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # Add feature dimension and mask this
            h = h_next * mask + hidden[0] * (1 - mask)
            c = c_next * mask + hidden[1] * (1 - mask)
        else:
            h = h_next
            c = c_next

        return h, c
