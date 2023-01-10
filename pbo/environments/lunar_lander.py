from typing import Tuple
from functools import partial
from gymnasium.wrappers.monitoring import video_recorder
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from pbo.networks.base_q import BaseQ


class LunarLanderEnv:
    def __init__(self, env_key: jax.random.PRNGKeyArray, gamma: float) -> None:
        self.n_actions = 4
        self.gamma = gamma
        self.reset_key = env_key
        self.actions_on_max = jnp.array([[0], [1], [2], [3]])

        self.env = gym.make("LunarLander-v2", render_mode="rgb_array")

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

    @partial(jax.jit, static_argnames=("self", "q"))
    def jitted_best_action_multi_head(self, q: BaseQ, q_params: hk.Params, state: jnp.ndarray) -> jnp.ndarray:
        state_repeat = jnp.repeat(state.reshape((1, 8)), self.actions_on_max.shape[0], axis=0)

        return self.actions_on_max[q(q_params, state_repeat, self.actions_on_max)[:, -1].argmax()]

    def evaluate_and_record(self, q: BaseQ, q_params: hk.Params, horizon: int, video_path: str) -> float:
        vid = video_recorder.VideoRecorder(
            self.env, path=f"experiments/lunar_lander/figures/{video_path}.mp4", disable_logger=True
        )
        cumulative_reward = jnp.array([0])
        discount = 1
        absorbing = jnp.array([False])
        self.reset()

        while not absorbing[0] and self.n_steps < horizon:
            self.env.render()
            vid.capture_frame()

            action = self.jitted_best_action(q, q_params, self.state)
            _, reward, absorbing, _ = self.step(action)

            cumulative_reward += discount * reward
            discount *= self.gamma

        self.env.close()
        vid.close()

        return np.array(cumulative_reward)[0]

    def evaluate_(self, q: BaseQ, q_params: hk.Params, horizon: int) -> float:
        cumulative_reward = jnp.array([0])
        discount = 1
        absorbing = jnp.array([False])
        self.reset()

        while not absorbing[0] and self.n_steps < horizon:
            action = self.jitted_best_action(q, q_params, self.state)
            _, reward, absorbing, _ = self.step(action)

            cumulative_reward += discount * reward
            discount *= self.gamma

        return np.array(cumulative_reward)[0]

    def evaluate(
        self, q: BaseQ, q_params: hk.Params, horizon: int, n_simulations: int, video_path: str = None
    ) -> float:
        rewards = np.zeros(n_simulations)

        if video_path is None:
            rewards[0] = self.evaluate_(q, q_params, horizon)
        else:
            rewards[0] = self.evaluate_and_record(q, q_params, horizon, video_path)

        for idx_simulation in range(1, n_simulations):
            rewards[idx_simulation] = self.evaluate_(q, q_params, horizon)

        return rewards.mean()
