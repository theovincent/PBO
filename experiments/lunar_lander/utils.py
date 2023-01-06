import os
import shutil
from functools import partial
from tqdm import tqdm
import jax
import jax.numpy as jnp
import haiku as hk

from pbo.environments.lunar_lander import LunarLanderEnv
from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.networks.base_pbo import BasePBO
from pbo.networks.base_q import BaseQ


def define_environment(env_key: jax.random.PRNGKeyArray, gamma: float) -> LunarLanderEnv:
    env = LunarLanderEnv(env_key, gamma)

    return env


def create_experiment_folders(experiment_name):
    if not os.path.exists(f"experiments/lunar_lander/figures/{experiment_name}"):
        os.makedirs(f"experiments/lunar_lander/figures/{experiment_name}/")
        shutil.copyfile(
            "experiments/lunar_lander/parameters.json",
            f"experiments/lunar_lander/figures/{experiment_name}/parameters.json",
        )

        os.mkdir(f"experiments/lunar_lander/figures/{experiment_name}/DQN/")
        os.mkdir(f"experiments/lunar_lander/figures/{experiment_name}/PBO_linear/")
        os.mkdir(f"experiments/lunar_lander/figures/{experiment_name}/PBO_deep/")
        os.mkdir(f"experiments/lunar_lander/figures/{experiment_name}/IDQN/")


def collect_random_samples(
    env: LunarLanderEnv, replay_buffer: ReplayBuffer, n_initial_samples: int, horizon: int
) -> None:
    env.reset()

    for _ in tqdm(range(n_initial_samples)):
        state = env.state

        env.sample_key, key = jax.random.split(env.sample_key)
        action = jax.random.choice(key, env.actions_on_max)
        next_state, reward, absorbing, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or env.n_steps > horizon:
            env.reset()


def collect_samples(
    env: LunarLanderEnv,
    replay_buffer: ReplayBuffer,
    q: BaseQ,
    q_params: hk.Params,
    n_steps: int,
    horizon: int,
    epsilon: float,
) -> None:
    for _ in range(n_steps):
        state = env.state

        env.sample_key, key = jax.random.split(env.sample_key)
        if jax.random.uniform(key) > epsilon:
            action = env.jitted_best_action(q, q_params, state)
        else:
            action = jax.random.choice(key, env.actions_on_max)

        next_state, reward, absorbing, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or env.n_steps >= horizon:
            env.reset()


def collect_samples_multi_head(
    env: LunarLanderEnv,
    replay_buffer: ReplayBuffer,
    q: BaseQ,
    q_params: hk.Params,
    n_steps: int,
    horizon: int,
    epsilon: float,
) -> None:
    for _ in range(n_steps):
        state = env.state

        env.sample_key, key = jax.random.split(env.sample_key)
        if jax.random.uniform(key) > epsilon:
            action = env.jitted_best_action_multi_head(q, q_params, state)
        else:
            action = jax.random.choice(key, env.actions_on_max)

        next_state, reward, absorbing, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or env.n_steps >= horizon:
            env.reset()


@partial(jax.jit, static_argnames=("pbo", "n_iterations"))
def iterated_q(pbo: BasePBO, pbo_params: hk.Params, q_weights: jnp.ndarray, n_iterations: int) -> jnp.ndarray:
    iterated_q_weights = q_weights

    for _ in range(n_iterations):
        iterated_q_weights = pbo(pbo_params, iterated_q_weights.reshape((1, -1)))[0]

    return iterated_q_weights
