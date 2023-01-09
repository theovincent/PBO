import jax

from pbo.environments.linear_quadratic import LinearQuadraticEnv


def define_environment(env_key: jax.random.PRNGKeyArray, max_init_state: float) -> LinearQuadraticEnv:
    env = LinearQuadraticEnv(env_key, max_init_state)

    return env
