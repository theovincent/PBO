from tqdm import tqdm
import numpy as np
from argparse import Namespace
import jax

from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.networks.base_pbo import BasePBO


def train(
    sample_key: jax.random.PRNGKey,
    environment_path: str,
    args: Namespace,
    p: dict,
    pbo: BasePBO,
    replay_buffer: ReplayBuffer,
) -> None:
    n_online_updates_per_iteraion = p["profqi_n_fitting_steps"] * replay_buffer.len // replay_buffer.batch_size
    losses = np.ones((p["profqi_n_training_steps"], n_online_updates_per_iteraion)) * np.nan

    for idx_trainin_step in tqdm(range(p["profqi_n_training_steps"])):
        pbo.update_target_params(0)

        for step in range(n_online_updates_per_iteraion):
            sample_key, key = jax.random.split(sample_key)
            # 1 so that the weights are not updated since n_training_steps_per_current_weight_update is large
            losses[idx_trainin_step, step] = pbo.update_online_params(1, replay_buffer, key)

    pbo.save(f"{environment_path}/{args.bellman_iterations_scope}_P_{args.seed}")
    np.save(f"{environment_path}/{args.bellman_iterations_scope}_L_{args.seed}", losses)
