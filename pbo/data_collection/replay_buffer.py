import numpy as np

import torch


class ReplayBuffer:
    def __init__(self) -> None:
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []

    def __len__(self) -> int:
        return len(self.state)

    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray) -> None:
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)

    def cast_to_tensor(self) -> None:
        self.state = torch.Tensor(np.array(self.state))
        self.action = torch.Tensor(np.array(self.action))
        self.reward = torch.Tensor(np.array(self.reward))
        self.next_state = torch.Tensor(np.array(self.next_state))
