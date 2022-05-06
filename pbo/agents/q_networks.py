from typing import OrderedDict
import numpy as np

import torch.nn as nn
import torch


class QFullyConnectedNet(nn.Module):
    def __init__(
        self, layer_dimension: int, random_range: float, action_range: float, n_discretisation_step_action: int
    ) -> None:
        super(QFullyConnectedNet, self).__init__()
        self.random_range = random_range
        self.action_range = action_range
        self.n_discretisation_step_action = n_discretisation_step_action

        self.network = nn.Sequential(
            OrderedDict(
                [
                    ("linear_1", nn.Linear(2, layer_dimension)),
                    ("relu", nn.ReLU()),
                    ("linear_2", nn.Linear(layer_dimension, 1)),
                ]
            )
        )

        self.linear_1_weight_shape = self.network.linear_1.weight.shape
        self.linear_1_bias_shape = self.network.linear_1.bias.shape
        self.linear_2_weight_shape = self.network.linear_2.weight.shape
        self.linear_2_bias_shape = self.network.linear_2.bias.shape

        self.q_weights_dimensions = (
            np.prod(self.linear_1_weight_shape)
            + np.prod(self.linear_1_bias_shape)
            + np.prod(self.linear_2_weight_shape)
            + np.prod(self.linear_2_bias_shape)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        stacked_state_action = torch.hstack((state, action))
        return self.network(stacked_state_action)

    def get_random_weights(self) -> torch.Tensor:
        return torch.FloatTensor(self.q_weights_dimensions).uniform_(-self.random_range, self.random_range)

    def set_weights(self, weights: torch.Tensor) -> None:
        current_index = 0

        weight_dimension = np.prod(self.linear_1_weight_shape)
        self.network.linear_1.weight.data = weights[current_index : current_index + weight_dimension].reshape(
            self.linear_1_weight_shape
        )
        current_index += weight_dimension

        weight_dimension = np.prod(self.linear_1_bias_shape)
        self.network.linear_1.bias.data = weights[current_index : current_index + weight_dimension].reshape(
            self.linear_1_bias_shape
        )
        current_index += weight_dimension

        weight_dimension = np.prod(self.linear_2_weight_shape)
        self.network.linear_2.weight.data = weights[current_index : current_index + weight_dimension].reshape(
            self.linear_2_weight_shape
        )
        current_index += weight_dimension

        weight_dimension = np.prod(self.linear_2_bias_shape)
        self.network.linear_2.bias.data = weights[current_index : current_index + weight_dimension].reshape(
            self.linear_2_bias_shape
        )
        current_index += weight_dimension

        assert (
            current_index == self.q_weights_dimensions
        ), f"Miss match between currend index: {current_index} and q weights dimension: {self.q_weights_dimensions}."

    def max_value(self, state: torch.Tensor) -> torch.Tensor:
        discrete_actions = torch.linspace(
            -self.action_range, self.action_range, steps=self.n_discretisation_step_action
        ).reshape((-1, 1))

        max_value_batch = torch.zeros(state.shape[0])

        for idx_s, s in enumerate(state):
            max_value_batch[idx_s] = self(
                s.repeat(self.n_discretisation_step_action).reshape((-1, 1)), discrete_actions
            ).max()

        return max_value_batch.reshape((-1, 1))
