import time
import matplotlib.pyplot as plt
import jax.numpy as jnp
from IPython.display import clear_output

import gym


class PongEnv:
    def __init__(self, env_seed: int) -> None:
        self.gym_env = gym.make("PongDeterministic-v4")
        self.env_seed = env_seed

    def reset(self, n_frames: int = 15) -> jnp.ndarray:
        self.state, self.information = self.gym_env.reset(seed=int(self.env_seed[0]))
        self.state = jnp.array(self.state)

        for _ in range(n_frames):
            self.state, _, _, _ = self.step(jnp.array([0]))

        return self.state

    def step(self, action: jnp.ndarray) -> tuple:
        self.state, reward, absorbing, _, self.information = self.gym_env.step(action[0])

        return jnp.array(self.state), jnp.array([reward]), jnp.array([absorbing]), {}

    def render(self) -> None:
        clear_output(wait=True)
        plt.imshow(self.state)
        plt.show()
        time.sleep(0.1)

    def close(self) -> None:
        clear_output()
