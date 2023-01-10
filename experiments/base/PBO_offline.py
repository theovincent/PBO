from argparse import Namespace
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp

from pbo.sample_collection.dataloader import SampleDataLoader
from pbo.weights_collection.dataloader import WeightsDataLoader
from pbo.networks.base_q import BaseQ
from pbo.networks.learnable_pbo import LinearPBO, MaxLinearPBO, CustomLinearPBO, DeepPBO
from pbo.utils.params import save_params


def train(
    environment_name: str,
    args: Namespace,
    q: BaseQ,
    p: dict,
    pbo_key: jax.random.PRNGKeyArray,
    data_loader_samples: SampleDataLoader,
    data_loader_weights: WeightsDataLoader,
    n_actions: int = None,
) -> None:
    learning_rate = {
        "first": p["starting_lr_pbo"],
        "last": p["ending_lr_pbo"],
        "duration": p["training_steps_pbo"]
        * p["fitting_steps_pbo"]
        * data_loader_samples.replay_buffer.len
        // p["batch_size_samples"],
    }
    if args.architecture == "linear":
        pbo = LinearPBO(
            q=q,
            max_bellman_iterations=args.max_bellman_iterations,
            add_infinity=True,
            network_key=pbo_key,
            learning_rate=learning_rate,
            initial_weight_std=p["initial_weight_std"],
        )
    elif args.architecture == "max_linear":
        pbo = MaxLinearPBO(
            q=q,
            max_bellman_iterations=args.max_bellman_iterations,
            network_key=pbo_key,
            learning_rate=learning_rate,
            n_actions=n_actions,
            initial_weight_std=p["initial_weight_std"],
        )
    elif args.architecture == "custom_linear":
        pbo = CustomLinearPBO(
            q=q,
            max_bellman_iterations=args.max_bellman_iterations,
            network_key=pbo_key,
            learning_rate=learning_rate,
            initial_weight_std=p["initial_weight_std"],
        )
    elif args.architecture == "deep":
        pbo = DeepPBO(
            q=q,
            max_bellman_iterations=args.max_bellman_iterations,
            network_key=pbo_key,
            layers_dimension=p["pbo_layers_dimension"],
            learning_rate=learning_rate,
            initial_weight_std=p["initial_weight_std"],
            conv=args.conv,
        )
    importance_iteration = jnp.ones(args.max_bellman_iterations + 1)

    l2_losses = np.ones((p["training_steps_pbo"], p["fitting_steps_pbo"])) * np.nan

    for training_step in tqdm(range(p["training_steps_pbo"])):
        params_target = pbo.params

        for fitting_step in tqdm(range(p["fitting_steps_pbo"]), leave=False):
            cumulative_l2_loss = 0

            data_loader_weights.shuffle()
            for batch_weights in data_loader_weights:
                data_loader_samples.shuffle()
                for batch_samples in data_loader_samples:
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
