from typing import Tuple
from functools import partial
import gymnasium as gym
import jax
import jax.numpy as jnp
import haiku as hk

from pbo.networks.base_q import BaseQ


class LunarLanderEnv:
    def __init__(self, env_key: jax.random.PRNGKeyArray, gamma: float) -> None:
        self.n_actions = 4
        self.gamma = gamma
        self.reset_key, self.sample_key = jax.random.split(env_key)
        self.actions_on_max = jnp.array([[0], [1], [2], [3]])

        self.env = gym.make("LunarLander-v2")

    def reset(self) -> jnp.ndarray:
        self.reset_key, key = jax.random.split(self.reset_key)
        self.state, _ = self.env.reset(seed=int(key[0]))
        self.n_steps = 0

        return jnp.array(self.state)

    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        self.state, reward, absorbing, _, info = self.env.step(int(action[0]))
        self.n_steps += 1

        return jnp.array(self.state), jnp.array([reward]), jnp.array([absorbing]), info

    @partial(jax.jit, static_argnames=("self", "q"))
    def jitted_best_action(self, q: BaseQ, q_params: hk.Params, state: jnp.ndarray) -> jnp.ndarray:
        state_repeat = jnp.repeat(state.reshape((1, 8)), self.actions_on_max.shape[0], axis=0)

        return self.actions_on_max[q(q_params, state_repeat, self.actions_on_max).argmax()]
