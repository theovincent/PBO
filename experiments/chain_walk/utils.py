import jax

from pbo.environments.chain_walk import ChainWalkEnv


def define_environment(
    env_key: jax.random.PRNGKeyArray, n_states: int, sucess_probability: float, gamma: float
) -> ChainWalkEnv:
    env = ChainWalkEnv(env_key, n_states, sucess_probability, gamma)

    return env
