import sys
import argparse
import json
import jax

from experiments.base.parser import addparse


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train a PBO on LQR.")
    addparse(parser, seed=True, architecture=True, validation_bellman_iterations=True)
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    print(
        f"Training a {args.architecture} PBO on LQR with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}..."
    )
    p = json.load(open(f"experiments/lqr/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.lqr.utils import define_q, define_data_loader_samples, generate_keys
    from pbo.weights_collection.weights_buffer import WeightsBuffer
    from pbo.weights_collection.dataloader import WeightsDataLoader
    from experiments.base.PBO_offline import train

    shuffle_key, q_key, pbo_key = generate_keys(args.seed)

    data_loader_samples = define_data_loader_samples(
        p["n_discrete_states"] * p["n_discrete_actions"], args.experiment_name, p["batch_size_samples"], shuffle_key
    )
    q = define_q(p["n_actions_on_max"], p["max_action_on_max"], q_key)

    weights_buffer = WeightsBuffer()
    weights_buffer.add(q.to_weights(q.params))

    q_random = define_q(p["n_actions_on_max"], p["max_action_on_max"], q_key, zero_initializer=False)
    # Add random weights
    while len(weights_buffer) < p["n_weights"]:
        weights = q_random.random_init_weights()
        weights_buffer.add(weights)

    weights_buffer.cast_to_jax_array()
    data_loader_weights = WeightsDataLoader(weights_buffer, p["batch_size_weights"], shuffle_key)

    train("lqr", args, q, p, pbo_key, data_loader_samples, data_loader_weights)
