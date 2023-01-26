from jax.random import KeyArray
import jax

from pbo.environments.linear_quadratic import LinearQuadraticEnv
from pbo.networks.learnable_q import LQRQ, LQRQVectorField
from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.sample_collection.dataloader import SampleDataLoader


def define_environment(env_key: jax.random.PRNGKeyArray, max_init_state: float) -> LinearQuadraticEnv:
    env = LinearQuadraticEnv(env_key, max_init_state)

    return env


def define_q(
    n_actions_on_max: int,
    max_action_on_max: float,
    m: float,
    key: jax.random.PRNGKeyArray,
    zero_initializer: bool = True,
    learning_rate: dict = None,
) -> LQRQ:
    if m is None:
        return LQRQ(
            n_actions_on_max=n_actions_on_max,
            max_action_on_max=max_action_on_max,
            network_key=key,
            zero_initializer=zero_initializer,
            learning_rate=learning_rate,
        )
    else:
        return LQRQVectorField(
            n_actions_on_max=n_actions_on_max,
            max_action_on_max=max_action_on_max,
            m=m,
            network_key=key,
            zero_initializer=zero_initializer,
            learning_rate=learning_rate,
        )


def define_data_loader_samples(n_samples, experiment_name: str, batch_size_samples, key) -> SampleDataLoader:
    replay_buffer = ReplayBuffer(n_samples)
    replay_buffer.load(f"experiments/lqr/figures/{experiment_name}/replay_buffer.npz")
    return SampleDataLoader(replay_buffer, batch_size_samples, key)


def generate_keys(seed: int) -> KeyArray:
    return jax.random.split(jax.random.PRNGKey(seed), 3)
