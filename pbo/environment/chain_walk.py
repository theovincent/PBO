import numpy as np

import jax
import jax.numpy as jnp


class ChainWalk:
    def __init__(
        self,
        n_states: int,
        sucess_probability: float,
        gamma: float,
        env_key: int,
        initial_state: jnp.ndarray = None,
    ) -> None:
        self.n_states = n_states
        self.sucess_probability = sucess_probability
        self.gamma = gamma
        self.next_state_key, self.reset_key = jax.random.split(env_key)

        self.initial_state = initial_state

        self.rewards = np.zeros(self.n_states)
        self.rewards[0] = 1
        self.rewards[-1] = 1

    def reset(self, state: jnp.ndarray = None) -> jnp.ndarray:
        if state is None:
            if self.initial_state is not None:
                self.state = self.initial_state
            else:
                self.state = jax.random.randint(self.reset_key, [1], minval=0, maxval=self.n_states)
        else:
            self.state = state

        return self.state

    def step(self, action: jnp.ndarray) -> tuple:
        if 0 < self.state[0] < self.n_states - 1:
            self.next_state_key, key = jax.random.split(self.next_state_key)
            if jax.random.uniform(key) <= self.sucess_probability:
                self.state = self.state - 1 if action[0] == 0 else self.state + 1
            reward = 0
        else:
            reward = 1

        absorbing = False if 0 < self.state[0] < self.n_states - 1 else True

        return self.state, jnp.array([reward]), absorbing, {}

    def apply_bellman_operator(self, q: np.ndarray) -> None:
        # left_shift_q = q on the first and last states
        left_shift_q = q.copy()
        left_shift_q[1:-1] = q[:-2]
        q[:, 0] = self.rewards + self.gamma * (
            self.sucess_probability * jnp.max(left_shift_q, axis=1) + (1 - self.sucess_probability) * jnp.max(q, axis=1)
        )

        # right_shift_q = q on the first and last states
        right_shift_q = q.copy()
        right_shift_q[1:-1] = q[2:, :]
        q[:, 1] = self.rewards + self.gamma * (
            self.sucess_probability * jnp.max(right_shift_q, axis=1)
            + (1 - self.sucess_probability) * jnp.max(q, axis=1)
        )

    def optimal_Q_mesh(self) -> jnp.ndarray:
        q = np.zeros((self.n_states, 2))
        changes = float("inf")

        while changes > 1e-9:
            old_q = q.copy()
            self.apply_bellman_operator(q)
            changes = np.linalg.norm(q - old_q)

        return jnp.array(q)
