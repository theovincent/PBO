import sys
import argparse
import json
import jax

from experiments.base.parser import addparse


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train a PBO on Chain Walk.")
    addparse(parser, seed=True, architecture=True, validation_bellman_iterations=True)
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    print(
        f"Training a {args.architecture} PBO on Chain Walk with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}..."
    )
    p = json.load(open(f"experiments/chain_walk/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.chain_walk.utils import define_environment, define_q, define_data_loader_samples, generate_keys
    from pbo.weights_collection.weights_buffer import WeightsBuffer
    from pbo.weights_collection.dataloader import WeightsDataLoader
    from experiments.base.PBO_offline import train

    shuffle_key, q_key, pbo_key = generate_keys(args.seed)

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["n_states"], p["sucess_probability"], p["gamma"])
    data_loader_samples = define_data_loader_samples(
        p["n_states"] * env.n_actions * p["n_repetitions"], args.experiment_name, p["batch_size_samples"], shuffle_key
    )
    q = define_q(p["n_states"], env.n_actions, p["gamma"], jax.random.PRNGKey(0))

    weights_buffer = WeightsBuffer()
    weights_buffer.add(q.to_weights(q.params))

    q_random = define_q(p["n_states"], env.n_actions, p["gamma"], q_key, zero_initializer=False)
    # Add random weights
    while len(weights_buffer) < p["n_weights"]:
        weights = q_random.random_init_weights()
        weights_buffer.add(weights)

    weights_buffer.cast_to_jax_array()
    data_loader_weights = WeightsDataLoader(weights_buffer, p["batch_size_weights"], shuffle_key)

    train("chain_walk", args, q, p, pbo_key, data_loader_samples, data_loader_weights, env.n_actions)
