import os
import shutil
from typing import Tuple
import numpy as np
import jax
import haiku as hk

from pbo.environments.lunar_lander import LunarLanderEnv
from pbo.sample_collection.replay_buffer import ReplayBuffer
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

    for _ in range(n_initial_samples):
        state = env.state

        env.sample_key, key = jax.random.split(env.sample_key)
        action = jax.random.choice(key, env.actions_on_max)
        next_state, reward, absorbing, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or env.n_steps > horizon:
            env.reset()


def collect_sample(
    env: LunarLanderEnv,
    replay_buffer: ReplayBuffer,
    q: BaseQ,
    q_params: hk.Params,
    n_samples: int,
    horizon: int,
    epsilon: float,
) -> None:
    env.reset()

    for _ in range(n_samples):
        state = env.state

        env.sample_key, key = jax.random.split(env.sample_key)
        if jax.random.uniform(key) > epsilon:
            action = q(q_params, state_to_repeat, env.actions_on_max).kind_of_an_argmax()
        else:
            action = jax.random.choice(key, env.actions_on_max)

        next_state, reward, absorbing, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or env.n_steps > horizon:
            env.reset()
