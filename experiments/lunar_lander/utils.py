from jax.random import KeyArray
from tqdm import tqdm
import jax
import jax.numpy as jnp
import haiku as hk

from pbo.environments.lunar_lander import LunarLanderEnv
from pbo.sample_collection.exploration import EpsilonGreedySchedule
from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.networks.base_q import BaseQ
from pbo.networks.learnable_q import FullyConnectedQ
from pbo.networks.learnable_multi_head_q import FullyConnectedMultiHeadQ


def define_environment(env_key: jax.random.PRNGKeyArray, gamma: float) -> LunarLanderEnv:
    env = LunarLanderEnv(env_key, gamma)

    return env


def collect_random_samples(
    env: LunarLanderEnv,
    sample_key: jax.random.PRNGKeyArray,
    replay_buffer: ReplayBuffer,
    n_initial_samples: int,
    horizon: int,
) -> None:
    env.reset()

    for _ in tqdm(range(n_initial_samples)):
        state = env.state

        sample_key, key = jax.random.split(sample_key)
        action = jax.random.choice(key, env.actions_on_max)
        next_state, reward, absorbing, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing[0] or env.n_steps >= horizon:
            env.reset()


def collect_samples(
    env: LunarLanderEnv,
    replay_buffer: ReplayBuffer,
    q: BaseQ,
    q_params: hk.Params,
    n_steps: int,
    horizon: int,
    exploration_schedule: EpsilonGreedySchedule,
) -> None:
    for _ in range(n_steps):
        state = env.state

        if exploration_schedule.explore():
            action = jax.random.choice(exploration_schedule.exploration_key, env.actions_on_max)
        else:
            action = env.jitted_best_action(q, q_params, state)

        next_state, reward, absorbing, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing[0] or env.n_steps >= horizon:
            env.reset()


def collect_samples_multi_head(
    env: LunarLanderEnv,
    replay_buffer: ReplayBuffer,
    q: BaseQ,
    q_params: hk.Params,
    n_steps: int,
    horizon: int,
    exploration_schedule: EpsilonGreedySchedule,
) -> None:
    for _ in range(n_steps):
        state = env.state

        if exploration_schedule.explore():
            action = jax.random.choice(exploration_schedule.exploration_key, env.actions_on_max)
        else:
            action = env.jitted_best_action_multi_head(q, q_params, state)

        next_state, reward, absorbing, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing[0] or env.n_steps >= horizon:
            env.reset()


def define_q(
    actions_on_max: jnp.ndarray,
    gamma: float,
    key: jax.random.PRNGKeyArray,
    layers_dimension: dict,
    learning_rate: dict = None,
) -> FullyConnectedQ:
    return FullyConnectedQ(
        state_dim=8,
        action_dim=1,
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
        state_dim=8,
        action_dim=1,
        actions_on_max=actions_on_max,
        gamma=gamma,
        network_key=key,
        layers_dimension=layers_dimension,
        zero_initializer=True,
        learning_rate=learning_rate,
    )


def generate_keys(seed: int) -> KeyArray:
    return jax.random.split(jax.random.PRNGKey(seed), 4)
