import torch.nn as nn
import torch

from pbo.agents.q_networks import QFullyConnectedNet


class BasePBO(nn.Module):
    def __init__(self, gamma: float, q_weights_dimensions: int) -> None:
        super(BasePBO, self).__init__()
        self.gamma = gamma
        self.q_weights_dimensions = q_weights_dimensions

    def define_criterion_and_optimizer(self) -> None:
        import torch.optim as optim

        self.criterion = torch.nn.L1Loss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001, momentum=0.9)

    def learn_on_batch(self, batch: dict, weights: torch.Tensor, q_network: QFullyConnectedNet) -> None:
        with torch.no_grad():
            q_network.set_weights(weights)
            target = batch["reward"] + self.gamma * q_network.max_value(batch["next_state"])

        q_network.set_weights(self.network(weights))

        loss = self.criterion(q_network(batch["state"], batch["action"]), target)

        loss.backward()
        self.optimizer.step()

    def get_fixed_point(self) -> torch.Tensor:
        raise NotImplementedError


class LinearPBONet(BasePBO):
    def __init__(self, gamma: float, q_weights_dimensions: int) -> None:
        super(LinearPBONet, self).__init__(gamma, q_weights_dimensions)

        self.network = nn.Linear(q_weights_dimensions, q_weights_dimensions)
        self.define_criterion_and_optimizer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_fixed_point(self) -> torch.Tensor:
        return -torch.inverse(self.network.weight - torch.eye(self.q_weights_dimensions)) @ self.network.bias
