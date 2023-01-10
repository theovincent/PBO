import sys
import argparse
import json
import jax
from experiments.base.parser import addparse


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train a PBO on Car-On-Hill.")
    addparse(parser, seed=True, architecture=True, validation_bellman_iterations=True)
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    print(
        f"Training a {args.architecture} PBO on Car-On-Hill with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}..."
    )
    if args.conv:
        print("PBO with convolutionnal layers.")
    p = json.load(open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.car_on_hill.utils import define_environment, define_q, define_data_loader_samples
    from pbo.weights_collection.weights_buffer import WeightsBuffer
    from pbo.weights_collection.dataloader import WeightsDataLoader
    from experiments.base.PBO_offline import train

    key = jax.random.PRNGKey(args.seed)
    shuffle_key, q_network_key, pbo_network_key = jax.random.split(key, 3)

    env, _, _, _, _ = define_environment(p["gamma"], p["n_states_x"], p["n_states_v"])
    data_loader_samples = define_data_loader_samples(
        p["n_random_samples"] + p["n_oriented_samples"], args.experiment_name, p["batch_size_samples"], shuffle_key
    )
    q = define_q(env.actions_on_max, p["gamma"], q_network_key, p["layers_dimension"])

    weights_buffer = WeightsBuffer()
    weights_buffer.add(q.to_weights(q.params))

    # Add random weights
    while len(weights_buffer) < p["n_weights"]:
        weights = q.random_init_weights()
        weights_buffer.add(weights)

    weights_buffer.cast_to_jax_array()
    data_loader_weights = WeightsDataLoader(weights_buffer, p["batch_size_weights"], shuffle_key)

    train("car_on_hill", args, q, p, pbo_network_key, data_loader_samples, data_loader_weights)
