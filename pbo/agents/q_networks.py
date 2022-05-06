from typing import OrderedDict
import numpy as np

import torch.nn as nn
import torch


class QFullyConnectedNet(nn.Module):
    def __init__(self, layer_dimension: int, random_range: float, action_range: float, n_discretisation_step_action: int) -> None:
        super().__init__()
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

        self.network_layers_dimension = OrderedDict()
        self.q_weights_dimensions = 0

        for name, param in self.network.named_parameters():
            if param.requires_grad:
                self.network_layers_dimension[name] = param.data.shape
                self.q_weights_dimensions += np.prod(param.data.shape)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        stacked_state_action = torch.hstack((state, action))
        return self.network(stacked_state_action)

    def get_random_weights(self) -> torch.Tensor:
        return torch.FloatTensor(self.q_weights_dimensions).uniform_(-self.random_range, self.random_range)

    def set_weights(self, weights: torch.Tensor) -> None:
        current_index = 0

        for name, shape in self.network_layers_dimension.items():
            name_layer, name_weights = name.split(".")
            weight_dimension = np.prod(shape)

            shapped_weights = weights[current_index: current_index + weight_dimension].reshape(shape)
            self.network.__getattr__(name_layer).__getattr__(name_weights) = shapped_weights

            current_index += weight_dimension

        assert current_index == self.q_weights_dimensions, "Not all the weights have been assigned"
    
    def max_value(self, state: torch.Tensor) -> torch.Tensor:
        discrete_actions = torch.linspace(- self.action_range, self.action_range, steps=self.n_discretisation_step_action)
        
        max_value_batch = torch.zeros(state.shape[0])

        for idx_s, s in enumerate(state):
            max_value_batch[idx_s] = self(s.repeat(self.n_discretisation_step_action), discrete_actions).max()

        return max_value_batch