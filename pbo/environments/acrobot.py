import gymnasium as gym
import jax

from pbo.environments.base import BaseEnv


class LunarLanderEnv(BaseEnv):
    def __init__(self, env_key: jax.random.PRNGKeyArray) -> None:
        super().__init__(env_key, gym.make("Acrobot-v1", render_mode="rgb_array"))
