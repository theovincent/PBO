import numpy as np

import jax.numpy as jnp


class ReplayBuffer:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

    def __len__(self) -> int:
        return len(self.states)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def cast_to_jax_array(self) -> None:
        self.states = jnp.array(self.states)
        self.actions = jnp.array(self.actions)
        self.rewards = jnp.array(self.rewards)
        self.next_states = jnp.array(self.next_states)
