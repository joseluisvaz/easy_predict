import torch
import math
from torch import nn, Tensor

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        )


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_p: float):
        super(SelfAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout_p, batch_first=True)

        self.mlp = nn.ModuleDict(
            dict(
                c_upproj=nn.Linear(embed_dim, 4 * embed_dim),
                c_downproj=nn.Linear(4 * embed_dim, embed_dim),
                act=NewGELU(),
                dropout=nn.Dropout(dropout_p),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_downproj(m.act(m.c_upproj(x))))  # MLP forward

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Interaction block for cross attention
        x: embedding
        cross_mask: mask for embedding, representing the keys
        """
        attention, _ = self.mha(query=x, key=x, value=x, key_padding_mask=mask)
        x = x + attention
        x = x + self.mlpf(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_p: float):
        super(CrossAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout_p, batch_first=True)

        self.mlp = nn.ModuleDict(
            dict(
                c_upproj=nn.Linear(embed_dim, 4 * embed_dim),
                c_downproj=nn.Linear(4 * embed_dim, embed_dim),
                act=NewGELU(),
                dropout=nn.Dropout(dropout_p),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_downproj(m.act(m.c_upproj(x))))  # MLP forward

    def forward(self, x: Tensor, cross_x: Tensor, cross_mask: Tensor) -> Tensor:
        """Interaction block for cross attention
        x: embedding
        cross_x: embedding to attend to
        cross_mask: mask for cross_x embedding, representing the keys
        """
        attention, _ = self.mha(query=x, key=cross_x, value=cross_x, key_padding_mask=cross_mask)
        x = x + attention
        x = x + self.mlpf(x)
        return x