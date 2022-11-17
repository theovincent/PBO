import os
import sys
import argparse
import json
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train a PBO on Car-On-Hill.")
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed of the training.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-b",
        "--max_bellman_iterations",
        help="Maximum number of Bellman iteration.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-a",
        "--architecture",
        help="Class of the PBO.",
        choices=["linear", "deep"],
        required=True,
    )
    args = parser.parse_args(argvs)
    print(
        f"Training a {args.architecture} PBO on Car-On-Hill with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}..."
    )
    p = json.load(open("experiments/car_on_hill/parameters.json"))  # p for parameters
    if not os.path.exists(f"experiments/car_on_hill/figures/data/PBO_{args.architecture}/"):
        os.makedirs(f"experiments/car_on_hill/figures/data/PBO_{args.architecture}/")

    from experiments.car_on_hill.utils import define_environment
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.weights_collection.weights_buffer import WeightsBuffer
    from pbo.sample_collection.dataloader import SampleDataLoader
    from pbo.weights_collection.dataloader import WeightsDataLoader
    from pbo.networks.learnable_q import FullyConnectedQ
    from pbo.networks.learnable_pbo import LinearPBO, DeepPBO
    from pbo.utils.params import save_params

    key = jax.random.PRNGKey(args.seed)
    shuffle_key, q_network_key, pbo_network_key = jax.random.split(key, 3)

    env, _, _, _, _ = define_environment(p["gamma"], p["n_states_x"], p["n_states_v"])

    replay_buffer = ReplayBuffer()
    replay_buffer.load("experiments/car_on_hill/figures/data/replay_buffer.npz")
    data_loader_samples = SampleDataLoader(replay_buffer, p["batch_size_samples"], shuffle_key)

    q = FullyConnectedQ(
        state_dim=2,
        action_dim=1,
        actions_on_max=env.actions_on_max,
        gamma=p["gamma"],
        network_key=q_network_key,
        layers_dimension=p["layers_dimension"],
        zero_initializer=True,
    )
    weights_buffer = WeightsBuffer()
    weights_buffer.add(q.to_weights(q.params))

    # Add random weights
    while len(weights_buffer) < p["n_weights"]:
        weights = q.random_init_weights()
        weights_buffer.add(weights)

    weights_buffer.cast_to_jax_array()
    data_loader_weights = WeightsDataLoader(weights_buffer, p["batch_size_weights"], shuffle_key)

    if args.architecture == "linear":
        add_infinity = False
        pbo = LinearPBO(
            q=q,
            max_bellman_iterations=args.max_bellman_iterations,
            add_infinity=add_infinity,
            network_key=pbo_network_key,
            learning_rate={
                "first": p["starting_lr_pbo"],
                "last": p["ending_lr_pbo"],
                "duration": p["training_steps"]
                * p["fitting_steps_pbo"]
                * len(replay_buffer)
                // p["batch_size_samples"],
            },
            initial_weight_std=p["initial_weight_std"],
        )
    else:
        add_infinity = False
        pbo = DeepPBO(
            q=q,
            max_bellman_iterations=args.max_bellman_iterations,
            network_key=pbo_network_key,
            learning_rate={
                "first": p["starting_lr_pbo"],
                "last": p["ending_lr_pbo"],
                "duration": p["training_steps"]
                * p["fitting_steps_pbo"]
                * len(replay_buffer)
                // p["batch_size_samples"],
            },
            initial_weight_std=p["initial_weight_std"],
        )
    importance_iteration = jnp.ones(args.max_bellman_iterations + 1)

    l2_losses = np.ones((p["training_steps"], p["fitting_steps_pbo"])) * np.nan

    for training_step in tqdm(range(p["training_steps"])):
        params_target = pbo.params

        for fitting_step in range(p["fitting_steps_pbo"]):
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
        f"experiments/car_on_hill/figures/data/PBO_{args.architecture}/{args.max_bellman_iterations}_P_{args.seed}",
        pbo.params,
    )
    np.save(
        f"experiments/car_on_hill/figures/data/PBO_{args.architecture}/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )
