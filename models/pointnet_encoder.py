# Modified from https://github.com/sshaoshuai/MTR

import typing as T

import torch
import torch.nn as nn


def _run_masked_function(
    function: T.Callable, input_tensor: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Helper to run a mapped function on a tensor with a valid mask"""
    features_valid = function(input_tensor[valid_mask])
    features = input_tensor.new_zeros(
        input_tensor.shape[:-1] + (features_valid.shape[-1],),
        dtype=features_valid.dtype,
    )
    features[valid_mask] = features_valid
    return features


def build_mlps(
    c_in: int,
    mlp_channels: T.List[int],
    ret_before_act: bool = False,
    without_norm: bool = False,
) -> nn.Sequential:
    """Helper function for point net MLP layers."""

    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()])
            else:
                layers.extend(
                    [
                        nn.Linear(c_in, mlp_channels[k], bias=False),
                        nn.BatchNorm1d(mlp_channels[k]),
                        nn.ReLU(),
                    ]
                )
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)


class PointNetPolylineEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_pre_layers: int = 1,
        out_channels: int = 128,
    ):
        super().__init__()
        self.pre_mlps = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False,
        )
        self.mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False,
        )

        self.out_mlps = build_mlps(
            c_in=hidden_dim,
            mlp_channels=[hidden_dim, out_channels],
            ret_before_act=True,
            without_norm=True,
        )

    def forward(
        self, polylines: torch.Tensor, polylines_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Dimension Naming:
            B: batch size
            P: number of polylines
            N: number of points in each polyline
            C: number of channels

        Args:
            polylines (B, P, N, C): polyline features
            polylines_mask (B, P, N): mask of the valid points in the polyline

        Returns:
            feature_buffers (B, P, C): reduction on the N dimension
        """

        assert len(polylines.shape) == 4
        assert len(polylines_mask.shape) == 3

        # Run MLP on each point in the polyline
        x = _run_masked_function(self.pre_mlps, polylines, polylines_mask)

        # Get global feature by performing a max-pooling on the N dimension
        x_pooled = x.max(dim=2)[0]
        x_pooled = x_pooled[:, :, None, :].repeat(1, 1, polylines.shape[2], 1)

        # Concatenate local and global pooled features and run another MLP on it
        x = torch.cat((x, x_pooled), dim=-1)
        x = _run_masked_function(self.mlps, x, polylines_mask)

        # Perform max-pooling on the N dimension to get a global feature with more
        # context and now run another MLP to extract more data from the global feature
        x_pooled = x.max(dim=2)[0]
        valid_mask = polylines_mask.sum(dim=-1) > 0  # (B, P)
        x_pooled = _run_masked_function(self.out_mlps, x_pooled, valid_mask)

        return x_pooled
