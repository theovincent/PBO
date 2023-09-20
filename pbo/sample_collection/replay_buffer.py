from typing import Dict, Type, Callable
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp


class ReplayBuffer:
    def __init__(
        self, max_size: int, batch_size: int, state_shape: list, state_dtype: Type, clipping: Callable
    ) -> None:
        self.max_size = max_size
        self.batch_size = batch_size
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.action_dtype = np.int8
        self.reward_dtype = np.float32
        self.absorbing_dtype = np.bool_
        self.clipping = clipping

        self.states = np.zeros((self.max_size,) + self.state_shape, dtype=self.state_dtype)
        self.actions = np.zeros(self.max_size, dtype=self.action_dtype)
        self.rewards = np.zeros(self.max_size, dtype=self.reward_dtype)
        self.next_states = np.zeros((self.max_size,) + self.state_shape, dtype=self.state_dtype)
        self.absorbings = np.zeros(self.max_size, dtype=self.absorbing_dtype)

        self.len = 0
        self.idx = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        absorbing: np.ndarray,
    ) -> None:
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = self.clipping(reward)
        self.next_states[self.idx] = next_state
        self.absorbings[self.idx] = absorbing

        self.idx += 1
        self.len = min(self.len + 1, self.max_size)
        if self.idx >= self.max_size:
            self.idx = 0

    def sample_random_batch(self, sample_key: jax.random.PRNGKeyArray) -> Dict[str, jnp.ndarray]:
        idxs = self.get_sample_indexes(sample_key, self.len)
        return self.create_batch(
            self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.absorbings[idxs]
        )

    @partial(jax.jit, static_argnames="self")
    def get_sample_indexes(self, key: jax.random.PRNGKeyArray, maxval: int) -> jnp.ndarray:
        return jax.random.randint(key, shape=(self.batch_size,), minval=0, maxval=maxval)

    @staticmethod
    @jax.jit
    def create_batch(
        states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, absorbings: np.ndarray
    ) -> Dict[str, jnp.ndarray]:
        return {
            "state": jnp.array(states, dtype=jnp.float32),
            "action": jnp.array(actions, dtype=jnp.int8),
            "reward": jnp.array(rewards, dtype=jnp.float32),
            "next_state": jnp.array(next_states, dtype=jnp.float32),
            "absorbing": jnp.array(absorbings, dtype=jnp.bool_),
        }
