import torch.nn as nn
import torch


class BasePBO(nn.Module):
    def __init__(self, gamma: float, q_weights_dimensions: int) -> None:
        self.gamma = gamma
        self.q_weights_dimensions = q_weights_dimensions
        self.network: nn.Module

    def define_criterion_and_optimizer(self) -> None:
        import torch.optim as optim

        self.criterion = torch.nn.L1Loss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001, momentum=0.9)

    def learn_on_batch(self, batch, weights, q_network) -> None:
        with torch.no_grad():
            q_network.set_weights(weights)
            target = batch.reward + self.gamma * q_network.max_value(batch.next_state)

        q_network.set_weights(self.network(weights))

        loss = self.criterion(q_network(batch.state, batch.action), target)

        loss.backward()
        self.optimizer.step()

    def get_fixed_point(self) -> torch.Tensor:
        raise NotImplementedError


class LinearPBONet(BasePBO):
    def __init__(self, gamma: float, q_weights_dimensions: int) -> None:
        super().__init__(gamma, q_weights_dimensions)

        self.network = nn.Linear(q_weights_dimensions, q_weights_dimensions)
        self.define_criterion_and_optimizer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_fixed_point(self) -> torch.Tensor:
        return -torch.inverse(self.network.weight - torch.eye(self.q_weights_dimensions)) @ self.network.bias
