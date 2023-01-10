import jax

from pbo.environments.chain_walk import ChainWalkEnv
from pbo.networks.learnable_q import TableQ
from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.sample_collection.dataloader import SampleDataLoader


def define_environment(
    env_key: jax.random.PRNGKeyArray, n_states: int, sucess_probability: float, gamma: float
) -> ChainWalkEnv:
    env = ChainWalkEnv(env_key, n_states, sucess_probability, gamma)

    return env


def define_q(
    n_states: int,
    n_actions: int,
    gamma: float,
    key: jax.random.PRNGKeyArray,
    zero_initializer: bool = True,
    learning_rate: dict = None,
) -> TableQ:
    return TableQ(
        n_states=n_states,
        n_actions=n_actions,
        gamma=gamma,
        network_key=key,
        zero_initializer=zero_initializer,
        learning_rate=learning_rate,
    )


def define_data_loader_samples(n_samples, experiment_name: str, batch_size_samples, key) -> SampleDataLoader:
    replay_buffer = ReplayBuffer(n_samples)
    replay_buffer.load(f"experiments/chain_walk/figures/{experiment_name}/replay_buffer.npz")
    return SampleDataLoader(replay_buffer, batch_size_samples, key)
