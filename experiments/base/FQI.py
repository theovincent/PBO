from tqdm import tqdm
import numpy as np
from argparse import Namespace
import jax

from pbo.environments.base import BaseEnv
from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.networks.base_q import BaseQ
from pbo.utils.params import save_pickled_data


def train(
    sample_key: jax.random.PRNGKey,
    environment_path: str,
    args: Namespace,
    p: dict,
    q: BaseQ,
    env: BaseEnv,
    replay_buffer: ReplayBuffer,
) -> None:
    env.collect_random_samples(sample_key, replay_buffer, p["n_samples"], p["horizon"])

    losses = np.ones((p["n_bellman_iterations"], p["fitting_steps"])) * np.nan
    iterated_params = {}
    iterated_params[0] = q.params

    for bellman_iteration in tqdm(range(p["n_bellman_iterations"])):
        q.update_target_params(0)

        for step in range(p["fitting_steps"]):
            sample_key, key = jax.random.split(sample_key)
            losses[bellman_iteration, step] = q.update_online_params(0, replay_buffer, key)

        iterated_params[bellman_iteration] = q.params

    save_pickled_data(f"{environment_path}/P_{args.seed}", iterated_params)
    np.save(f"{environment_path}/L_{args.seed}.npy", losses)
