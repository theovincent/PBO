import jax

from pbo.environments.bicycle import BicycleEnv


def define_environment(env_key: jax.random.PRNGKeyArray, gamma: float) -> BicycleEnv:
    env = BicycleEnv(env_key, gamma)

    return env
