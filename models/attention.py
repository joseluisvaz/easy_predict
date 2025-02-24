import math
import typing as T

import torch
from torch import Tensor, nn


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_p: float):
        super(SelfAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout_p, batch_first=True
        )

        self.mlp_block = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # Up projection
            NewGELU(),
            nn.Linear(4 * embed_dim, embed_dim),  # Down projection
            nn.Dropout(dropout_p),
        )

    def forward(
        self, x: Tensor, mask: Tensor, pos_embedding: T.Optional[Tensor] = None
    ) -> Tensor:
        """Interaction block for cross attention
        x: embedding
        cross_mask: mask for embedding, representing the keys
        """
        q = k = x + pos_embedding if pos_embedding is not None else x
        attention, _ = self.mha(query=q, key=k, value=x, key_padding_mask=mask)
        x = x + attention
        x = x + self.mlp_block(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_p: float):
        super(CrossAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout_p, batch_first=True
        )

        self.mlp_block = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # Up projection
            NewGELU(),
            nn.Linear(4 * embed_dim, embed_dim),  # Down projection
            nn.Dropout(dropout_p),
        )

    def forward(self, x: Tensor, cross_x: Tensor, cross_mask: Tensor) -> Tensor:
        """Interaction block for cross attention
        x: embedding
        cross_x: embedding to attend to
        cross_mask: mask for cross_x embedding, representing the keys
        """
        attention, _ = self.mha(
            query=x, key=cross_x, value=cross_x, key_padding_mask=cross_mask
        )
        x = x + attention
        x = x + self.mlp_block(x)
        return x
