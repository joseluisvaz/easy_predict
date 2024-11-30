from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[Callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Implementation for the dot-product operation for the attention mechanism.
    Args:
        query (torch.Tensor): size [batch, head, 1 (ego-entity), features]
        key (torch.Tensor): size [batch, head, entities, features]
        value (torch.Tensor): size [batch, head, entities, features]
        mask (Optional[torch.Tensor]): size [batch,  head], 0 (absence feature), 0 (ego-entity)
        dropout (Optional[Callable]): pytorch dropout module
    Returns:
        (torch.Tensor) scaled dot product attention
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


class MultiheadAttention(nn.Module):
    """
    Adapted from:
     https://github.com/eleurent/rl-agents/blob/fb59a48ca8058a14ee6197227b92ded63f85433c/rl_agents/configuration.py
    """

    def __init__(self, in_feats: int, per_head_feats: int, n_heads: int, dropout_factor: float) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.per_head_feats = per_head_feats
        self.n_heads = n_heads
        self.dropout_factor = dropout_factor

        total_features = per_head_feats * n_heads
        self.value_all = nn.Linear(self.in_feats, total_features, bias=False)
        self.key_all = nn.Linear(self.in_feats, total_features, bias=False)
        self.query_all = nn.Linear(self.in_feats, total_features, bias=False)
        self.attention_combine = nn.Linear(total_features, in_feats, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *args: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation of the multiheaded attention layer, with masked tokens.
        Args:
            query: size [n_batch, query_size, feature_size]
            keys: size [n_batch, key_size, feature_size]
            values: size [n_batch, key_size, feature_size]
            mask: size [n_batch, n_agent]
        Returns:
            (torch.Tensor) size [n_batch, n_agents, feature_size]
        """

        total_features = self.n_heads * self.per_head_feats

        _, query_size, _ = query.shape
        n_batch, key_size, _ = keys.shape

        key_all = self.key_all(keys).view(n_batch, key_size, self.n_heads, self.per_head_feats)
        value_all = self.value_all(values).view(n_batch, key_size, self.n_heads, self.per_head_feats)
        query_all = self.query_all(query).view(n_batch, query_size, self.n_heads, self.per_head_feats)

        # Dimensions: Batch, head, agents, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_all = query_all.permute(0, 2, 1, 3)

        if mask is not None:
            mask = mask.view((n_batch, 1, 1, key_size)).repeat((1, self.n_heads, 1, 1))
        output, attention_matrix = attention(query_all, key_all, value_all, mask, nn.Dropout(self.dropout_factor))

        output = self.attention_combine(output.reshape((n_batch, query_size, total_features)))
        return output, attention_matrix
