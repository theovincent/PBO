import os
from typing import Tuple, Union
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
from tqdm import tqdm
from gymnasium import Env
from gymnasium.wrappers.monitoring import video_recorder

from pbo.networks.base_q import BaseQ
from pbo.networks.base_pbo import BasePBO
from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.sample_collection.exploration import EpsilonGreedySchedule


class BaseEnv:
    def __init__(self, env_key: jax.random.PRNGKeyArray, env: Env) -> None:
        self.reset_key = env_key
        self.env = env
        self.state_shape = env.observation_space.shape
        self.n_actions = self.env.action_space.n

    def reset(self) -> jnp.ndarray:
        self.reset_key, key = jax.random.split(self.reset_key)
        self.state, _ = self.env.reset(seed=int(key[0]))
        self.n_steps = 0

        return self.state

    def step(self, action: jnp.int8) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        self.state, reward, absorbing, _, info = self.env.step(int(action))
        self.n_steps += 1

        return self.state, reward, absorbing, info

    def collect_random_samples(
        self, sample_key: jax.random.PRNGKeyArray, replay_buffer: ReplayBuffer, n_samples: int, horizon: int
    ) -> None:
        self.reset()

        for _ in tqdm(range(n_samples)):
            state = self.state

            sample_key, key = jax.random.split(sample_key)
            action = jax.random.choice(key, jnp.arange(self.n_actions))
            next_state, reward, absorbing, _ = self.step(action)

            replay_buffer.add(state, action, reward, next_state, absorbing)

            if absorbing or self.n_steps >= horizon:
                self.reset()

    def collect_one_sample(
        self,
        q_or_pbo: Union[BaseQ, BasePBO],
        q_or_pbo_params: FrozenDict,
        horizon: int,
        replay_buffer: ReplayBuffer,
        exploration_schedule: EpsilonGreedySchedule,
    ) -> Tuple[float, bool]:
        state = self.state

        if exploration_schedule.explore():
            action = q_or_pbo.random_action(exploration_schedule.key)
        else:
            action = q_or_pbo.best_action(q_or_pbo_params, self.state, exploration_schedule.key)

        next_state, reward, absorbing, _ = self.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or self.n_steps >= horizon:
            self.reset()

        return reward, absorbing or self.n_steps == 0

    def evaluate(
        self,
        q_or_pbo: Union[BaseQ, BasePBO],
        q_or_pbo_params: FrozenDict,
        horizon: int,
        eps_eval: float,
        exploration_key: jax.random.PRNGKey,
        video_path: Union[str, None],
    ) -> float:
        if video_path is not None:
            video = video_recorder.VideoRecorder(self.env, path=f"{video_path}.mp4", disable_logger=True)
        sun_reward = 0
        absorbing = False
        self.reset()

        while not absorbing and self.n_steps < horizon:
            if video_path is not None:
                self.env.render()
                video.capture_frame()

            exploration_key, key = jax.random.split(exploration_key)
            if jax.random.uniform(key) < eps_eval:
                action = q_or_pbo.random_action(key)
            else:
                action = q_or_pbo.best_action(q_or_pbo_params, self.state, key)

            _, reward, absorbing, _ = self.step(action)

            sun_reward += reward

        if video_path is not None:
            video.close()
            os.remove(f"{video_path}.meta.json")

        return sun_reward
