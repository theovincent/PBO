from argparse import Namespace
from tqdm import tqdm
import numpy as np
import jax

from pbo.sample_collection.exploration import EpsilonGreedySchedule
from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.networks.base_q import BaseQ
from pbo.utils.params import save_params


def train(
    environment_name: str,
    args: Namespace,
    q: BaseQ,
    p: dict,
    exploration_key: jax.random.PRNGKeyArray,
    sample_key: jax.random.PRNGKeyArray,
    replay_buffer: ReplayBuffer,
    collect_samples: function,
    env,
) -> None:
    epsilon_schedule = EpsilonGreedySchedule(
        p["starting_eps_dqn"],
        p["ending_eps_dqn"],
        args.max_bellman_iterations * p["fitting_steps_dqn"] * p["steps_per_update"],
        exploration_key,
    )

    l2_losses = np.ones((args.max_bellman_iterations, p["fitting_steps_dqn"])) * np.nan
    iterated_params = {}
    iterated_params["0"] = q.params

    for bellman_iteration in tqdm(range(1, args.max_bellman_iterations + 1)):
        params_target = q.params

        for update in tqdm(range(p["fitting_steps_dqn"]), leave=False):
            collect_samples(env, replay_buffer, q, q.params, p["steps_per_update"], p["horizon"], epsilon_schedule)

            sample_key, key = jax.random.split(sample_key)
            batch_samples = replay_buffer.sample_random_batch(key, p["batch_size_samples"])

            q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
                q.params, params_target, q.optimizer_state, batch_samples
            )

            l2_losses[bellman_iteration - 1, update] = l2_loss

        iterated_params[f"{bellman_iteration}"] = q.params

    save_params(
        f"experiments/{environment_name}/figures/{args.experiment_name}/DQN/{args.max_bellman_iterations}_P_{args.seed}",
        iterated_params,
    )
    np.save(
        f"experiments/{environment_name}/figures/{args.experiment_name}/DQN/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )
