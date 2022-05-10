import numpy as np

import torch


class ReplayBuffer:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

    def __len__(self) -> int:
        return len(self.states)

    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def cast_to_tensor(self) -> None:
        self.states = torch.Tensor(np.array(self.states))
        self.actions = torch.Tensor(np.array(self.actions))
        self.rewards = torch.Tensor(np.array(self.rewards))
        self.next_states = torch.Tensor(np.array(self.next_states))
