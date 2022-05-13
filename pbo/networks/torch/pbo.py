from typing import OrderedDict

import torch.nn as nn
import torch

from pbo.networks.torch.q import QFullyConnectedNet


class BasePBO(nn.Module):
    def __init__(self, gamma: float, q_weights_dimensions: int) -> None:
        super(BasePBO, self).__init__()
        self.gamma = gamma
        self.q_weights_dimensions = q_weights_dimensions
        self.criterion = torch.nn.L1Loss()

    def loss(self, batch: dict, weights: torch.Tensor, q_network: QFullyConnectedNet) -> torch.Tensor:
        with torch.no_grad():
            q_network.set_weights(weights)
            target = batch["reward"] + self.gamma * q_network.max_value(batch["next_state"])

        q_network.set_weights(self(weights))

        loss = self.criterion(q_network(batch["state"], batch["action"]), target)

        return loss

    def get_fixed_point(self) -> torch.Tensor:
        raise NotImplementedError


class LinearPBONet(BasePBO):
    def __init__(self, gamma: float, q_weights_dimensions: int) -> None:
        super(LinearPBONet, self).__init__(gamma, q_weights_dimensions)

        self.network = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(q_weights_dimensions, q_weights_dimensions)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_fixed_point(self) -> torch.Tensor:
        return (
            -torch.inverse(self.network.linear.weight.data - torch.eye(self.q_weights_dimensions))
            @ self.network.linear.bias.data
        )
