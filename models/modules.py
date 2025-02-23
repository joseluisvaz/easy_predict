import torch
from torch import nn, Tensor


class DynamicsLayer(nn.Module):
    def __init__(self, delta_t: float, max_acc: float, max_yaw_rate: float) -> None:
        nn.Module.__init__(self)

        self.n_state = 5
        self.n_input = 2

        self.delta_t = delta_t
        self.max_acc = max_acc
        self.max_yaw_rate = max_yaw_rate

    def get_state_from_features(self, features: Tensor) -> Tensor:
        # The first elements are x, y, c, s, v
        return features[..., :5]

    def activation_fn(self, _input: Tensor) -> Tensor:
        return torch.stack(
            (
                self.max_acc * torch.tanh(_input[..., 0]),
                self.max_yaw_rate * torch.tanh(_input[..., 1]),
            ),
            dim=-1,
        )

    def forward(self, state: Tensor, _input: Tensor) -> Tensor:
        accel = _input[..., 0]
        yaw_rate = _input[..., 1]

        state_dot = torch.zeros_like(state)

        state_dot[..., 0] = state[..., 4] * state[..., 2]
        state_dot[..., 1] = state[..., 4] * state[..., 3]
        state_dot[..., 2] = -yaw_rate * state[..., 3]
        state_dot[..., 3] = yaw_rate * state[..., 2]
        state_dot[..., 4] = accel

        return state + self.delta_t * state_dot
