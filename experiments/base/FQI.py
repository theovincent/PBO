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
    experiment_path: str,
    args: Namespace,
    p: dict,
    q: BaseQ,
    replay_buffer: ReplayBuffer,
) -> None:
    n_online_updates_per_iteraion = int(p["fqi_n_fitting_steps"] * replay_buffer.len / replay_buffer.batch_size)
    losses = np.ones((args.bellman_iterations_scope, n_online_updates_per_iteraion)) * np.nan
    iterated_params = {}
    iterated_params[0] = q.params

    for bellman_iteration in tqdm(range(args.bellman_iterations_scope)):
        q.update_target_params(0)

        for step in range(n_online_updates_per_iteraion):
            sample_key, key = jax.random.split(sample_key)
            losses[bellman_iteration, step] = q.update_online_params(0, replay_buffer, key)

        iterated_params[bellman_iteration + 1] = q.params

    save_pickled_data(f"{experiment_path}/{args.bellman_iterations_scope}_P_{args.seed}", iterated_params)
    np.save(f"{experiment_path}/{args.bellman_iterations_scope}_L_{args.seed}.npy", losses)
