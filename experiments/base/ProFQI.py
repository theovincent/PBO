from tqdm import tqdm
import numpy as np
from argparse import Namespace
import jax

from pbo.environments.base import BaseEnv
from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.networks.base_PBO import BasePBO


def train(
    sample_key: jax.random.PRNGKey,
    environment_path: str,
    args: Namespace,
    p: dict,
    pbo: BasePBO,
    env: BaseEnv,
    replay_buffer: ReplayBuffer,
) -> None:
    env.collect_random_samples(sample_key, replay_buffer, p["n_samples"], p["horizon"])

    losses = np.ones((p["n_bellman_iterations"], p["fitting_steps"])) * np.nan

    for bellman_iteration in tqdm(range(p["n_bellman_iterations"])):
        pbo.update_target_params()

        for step in range(p["fitting_steps"]):
            sample_key, key = jax.random.split(sample_key)
            losses[bellman_iteration, step] = pbo.update_online_params(replay_buffer, key)

    pbo.save(f"{environment_path}/P_{args.seed}.npy")
    np.save(f"{environment_path}/L_{args.seed}.npy", losses)
