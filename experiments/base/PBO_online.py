from argparse import Namespace
from functools import partial
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from pbo.sample_collection.exploration import EpsilonGreedySchedule
from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.weights_collection.dataloader import WeightsDataLoader
from pbo.networks.base_q import BaseQ
from pbo.networks.base_pbo import BasePBO
from pbo.networks.learnable_pbo import LinearPBO, DeepPBO
from pbo.utils.params import save_params


def train(
    environment_name: str,
    args: Namespace,
    q: BaseQ,
    p: dict,
    pbo_network_key: jax.random.PRNGKeyArray,
    exploration_key: jax.random.PRNGKeyArray,
    sample_key: jax.random.PRNGKeyArray,
    replay_buffer: ReplayBuffer,
    data_loader_weights: WeightsDataLoader,
    collect_samples: function,
    env,
) -> None:
    epsilon_schedule = EpsilonGreedySchedule(
        p["starting_eps_pbo"],
        p["ending_eps_pbo"],
        p["training_steps_pbo"] * p["fitting_updates_pbo"] * p["steps_per_update"],
        exploration_key,
    )
    learning_rate = {
        "first": p["starting_lr_pbo"],
        "last": p["ending_lr_pbo"],
        "duration": p["training_steps_pbo"] * p["fitting_updates_pbo"],
    }
    if args.architecture == "linear":
        pbo = LinearPBO(
            q=q,
            max_bellman_iterations=args.max_bellman_iterations,
            add_infinity=True,
            network_key=pbo_network_key,
            learning_rate=learning_rate,
            initial_weight_std=p["initial_weight_std"],
        )
    else:
        pbo = DeepPBO(
            q=q,
            max_bellman_iterations=args.max_bellman_iterations,
            network_key=pbo_network_key,
            layers_dimension=p["pbo_layers_dimension"],
            learning_rate=learning_rate,
            initial_weight_std=p["initial_weight_std"],
            conv=args.conv,
        )
    importance_iteration = jnp.ones(args.max_bellman_iterations + 1)

    l2_losses = np.ones((p["training_steps_pbo"], p["fitting_updates_pbo"])) * np.nan

    for training_step in tqdm(range(p["training_steps_pbo"])):
        params_target = pbo.params

        for fitting_step in tqdm(range(p["fitting_updates_pbo"]), leave=False):
            cumulative_l2_loss = 0
            q_weights_exploration = iterated_q(
                pbo, pbo.params, data_loader_weights.weights_buffer.weights[0], args.max_bellman_iterations
            )
            collect_samples(
                env,
                replay_buffer,
                q,
                q.to_params(q_weights_exploration),
                p["steps_per_update"],
                p["horizon"],
                epsilon_schedule,
            )

            sample_key, key = jax.random.split(sample_key)
            batch_samples = replay_buffer.sample_random_batch(key, p["batch_size_samples"])

            data_loader_weights.shuffle()
            for batch_weights in data_loader_weights:
                pbo.params, pbo.optimizer_state, l2_loss = pbo.learn_on_batch(
                    pbo.params,
                    params_target,
                    pbo.optimizer_state,
                    batch_weights,
                    batch_samples,
                    importance_iteration,
                )
                cumulative_l2_loss += l2_loss

            l2_losses[training_step, fitting_step] = cumulative_l2_loss

    save_params(
        f"experiments/{environment_name}/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_P_{args.seed}",
        pbo.params,
    )
    np.save(
        f"experiments/{environment_name}/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )


@partial(jax.jit, static_argnames=("pbo", "n_iterations"))
def iterated_q(pbo: BasePBO, pbo_params: hk.Params, q_weights: jnp.ndarray, n_iterations: int) -> jnp.ndarray:
    iterated_q_weights = q_weights

    for _ in range(n_iterations):
        iterated_q_weights = pbo(pbo_params, iterated_q_weights.reshape((1, -1)))[0]

    return iterated_q_weights
