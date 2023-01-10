import jax
from jax.random import KeyArray
import jax.numpy as jnp

from pbo.environments.bicycle import BicycleEnv
from pbo.networks.learnable_q import FullyConnectedQ
from pbo.networks.learnable_multi_head_q import FullyConnectedMultiHeadQ
from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.sample_collection.dataloader import SampleDataLoader


def define_environment(env_key: jax.random.PRNGKeyArray, gamma: float) -> BicycleEnv:
    env = BicycleEnv(env_key, gamma)

    return env


def define_q(
    actions_on_max: jnp.ndarray,
    gamma: float,
    key: jax.random.PRNGKeyArray,
    layers_dimension: dict,
    learning_rate: dict = None,
) -> FullyConnectedQ:
    return FullyConnectedQ(
        state_dim=4,
        action_dim=2,
        actions_on_max=actions_on_max,
        gamma=gamma,
        network_key=key,
        layers_dimension=layers_dimension,
        zero_initializer=True,
        learning_rate=learning_rate,
    )


def define_q_multi_head(
    n_heads: int,
    actions_on_max: jnp.ndarray,
    gamma: float,
    key: jax.random.PRNGKeyArray,
    layers_dimension: dict,
    learning_rate: dict = None,
) -> FullyConnectedMultiHeadQ:
    return FullyConnectedMultiHeadQ(
        n_heads=n_heads,
        state_dim=4,
        action_dim=2,
        actions_on_max=actions_on_max,
        gamma=gamma,
        network_key=key,
        layers_dimension=layers_dimension,
        zero_initializer=True,
        learning_rate=learning_rate,
    )


def define_data_loader_samples(n_samples, experiment_name: str, batch_size_samples, key) -> SampleDataLoader:
    replay_buffer = ReplayBuffer(n_samples)
    replay_buffer.load(f"experiments/bicycle/figures/{experiment_name}/replay_buffer.npz")
    return SampleDataLoader(replay_buffer, batch_size_samples, key)


def generate_keys(seed: int) -> KeyArray:
    return jax.random.split(jax.random.PRNGKey(seed), 3)
