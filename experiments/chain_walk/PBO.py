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

    parser = argparse.ArgumentParser("Train a PBO on Chain Walk.")
    parser.add_argument(
        "-e",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
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
        choices=["linear", "max_linear", "deep"],
        required=True,
    )
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    print(
        f"Training a {args.architecture} PBO on Chain Walk with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}..."
    )
    p = json.load(open(f"experiments/chain_walk/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.chain_walk.utils import define_environment
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.weights_collection.weights_buffer import WeightsBuffer
    from pbo.sample_collection.dataloader import SampleDataLoader
    from pbo.weights_collection.dataloader import WeightsDataLoader
    from pbo.networks.learnable_q import TableQ
    from pbo.networks.learnable_pbo import LinearPBO, DeepPBO, MaxLinearPBO
    from pbo.utils.params import save_params

    key = jax.random.PRNGKey(args.seed)
    shuffle_key, pbo_network_key = jax.random.split(key, 2)

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["n_states"], p["sucess_probability"], p["gamma"])

    replay_buffer = ReplayBuffer(p["n_states"] * env.n_actions * p["n_repetitions"])
    replay_buffer.load(f"experiments/chain_walk/figures/{args.experiment_name}/replay_buffer.npz")
    data_loader_samples = SampleDataLoader(replay_buffer, p["batch_size_samples"], shuffle_key)

    q = TableQ(
        n_states=p["n_states"],
        n_actions=env.n_actions,
        gamma=p["gamma"],
        network_key=jax.random.PRNGKey(0),
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
                "duration": p["training_steps_pbo"]
                * p["fitting_steps_pbo"]
                * replay_buffer.len
                // p["batch_size_samples"],
            },
            initial_weight_std=p["initial_weight_std"],
        )
    elif args.architecture == "max_linear":
        add_infinity = False
        pbo = MaxLinearPBO(
            q=q,
            max_bellman_iterations=args.max_bellman_iterations,
            network_key=pbo_network_key,
            learning_rate={
                "first": p["starting_lr_pbo"],
                "last": p["ending_lr_pbo"],
                "duration": p["training_steps_pbo"]
                * p["fitting_steps_pbo"]
                * replay_buffer.len
                // p["batch_size_samples"],
            },
            n_actions=env.n_actions,
            initial_weight_std=p["initial_weight_std"],
        )
    else:
        add_infinity = False
        pbo = DeepPBO(
            q=q,
            max_bellman_iterations=args.max_bellman_iterations,
            network_key=pbo_network_key,
            layers_dimension=p["pbo_layers_dimension"],
            learning_rate={
                "first": p["starting_lr_pbo"],
                "last": p["ending_lr_pbo"],
                "duration": p["training_steps_pbo"]
                * p["fitting_steps_pbo"]
                * replay_buffer.len
                // p["batch_size_samples"],
            },
            initial_weight_std=p["initial_weight_std"],
            conv=False,
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
        f"experiments/chain_walk/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_P_{args.seed}",
        pbo.params,
    )
    np.save(
        f"experiments/chain_walk/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )
