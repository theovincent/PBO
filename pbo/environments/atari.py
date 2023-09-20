"""
The environment is inspired from https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py
"""
from typing import Tuple, Dict
import gymnasium as gym
import numpy as np
import jax.numpy as jnp
from collections import deque
import cv2

from pbo.environments.base import BaseEnv
from pbo.environments.extract_pvn import extract_pvn
from pbo.environments.pvns import FEAUTURE_SIZE


class AtariEnv(BaseEnv):
    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name
        self.state_height, self.state_width = (84, 84)
        self.n_stacked_frames = 4
        self.n_skipped_frames = 4
        self.state_shape = (FEAUTURE_SIZE,)

        self.env = gym.make(
            f"ALE/{self.name}-v5",
            full_action_space=False,
            frameskip=1,
            repeat_action_probability=0.25,
            render_mode="rgb_array",
        ).env
        self.encoder = extract_pvn(self.name)

        self.n_actions = self.env.action_space.n
        self.original_state_height, self.original_state_width, _ = self.env.observation_space._shape
        self.screen_buffer = [
            np.empty((self.original_state_height, self.original_state_width), dtype=np.uint8),
            np.empty((self.original_state_height, self.original_state_width), dtype=np.uint8),
        ]

    @property
    def state(self) -> np.ndarray:
        return self.encoder(np.array(self.stacked_frames, ndmin=4).transpose((0, 2, 3, 1)) / 255.0).astype(np.float32)

    def reset(self) -> np.ndarray:
        self.env.reset()

        self.n_steps = 0

        self.env.ale.getScreenGrayscale(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)

        self.stacked_frames = deque(
            np.repeat(self.resize()[None, ...], self.n_stacked_frames, axis=0),
            maxlen=self.n_stacked_frames,
        )

        return self.state

    def step(self, action: jnp.int8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        reward = 0

        for idx_frame in range(self.n_skipped_frames):
            _, reward_, absorbing, _, _ = self.env.step(action)

            reward += reward_

            if idx_frame >= self.n_skipped_frames - 2:
                t = idx_frame - (self.n_skipped_frames - 2)
                self.env.ale.getScreenGrayscale(self.screen_buffer[t])

            if absorbing:
                break

        self.stacked_frames.append(self.pool_and_resize())

        self.n_steps += 1

        return self.state, reward, absorbing, _

    def pool_and_resize(self) -> np.ndarray:
        np.maximum(self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0])

        return self.resize()

    def resize(self):
        return np.asarray(
            cv2.resize(self.screen_buffer[0], (self.state_width, self.state_height), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )
