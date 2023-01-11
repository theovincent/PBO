import sys
import argparse
import json
import jax

from experiments.base.parser import addparse


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train a PBO on Lunar Lander.")
    addparse(parser, seed=True, architecture=True, validation_bellman_iterations=True)
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    print(
        f"Training a {args.architecture} PBO on Lunar Lander with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}..."
    )
    if args.conv:
        print("PBO with convolutionnal layers.")
    p = json.load(open(f"experiments/lunar_lander/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.lunar_lander.utils import (
        define_environment,
        define_q,
        collect_random_samples,
        collect_samples,
        generate_keys,
    )
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.weights_collection.weights_buffer import WeightsBuffer
    from pbo.weights_collection.dataloader import WeightsDataLoader
    from experiments.base.PBO_online import train

    sample_key, exploration_key, q_key, pbo_key = generate_keys(args.seed)
    shuffle_key = sample_key

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

    replay_buffer = ReplayBuffer(p["max_size"])
    collect_random_samples(env, replay_buffer, p["n_initial_samples"], p["horizon"])

    q = define_q(env.actions_on_max, p["gamma"], q_key, p["layers_dimension"])

    weights_buffer = WeightsBuffer()
    weights_buffer.add(q.to_weights(q.params))

    # Add random weights
    while len(weights_buffer) < p["n_weights"]:
        weights = q.random_init_weights()
        weights_buffer.add(weights)

    weights_buffer.cast_to_jax_array()
    data_loader_weights = WeightsDataLoader(weights_buffer, p["batch_size_weights"], shuffle_key)

    train(
        "lunar_lander",
        args,
        q,
        p,
        pbo_key,
        exploration_key,
        sample_key,
        replay_buffer,
        data_loader_weights,
        collect_samples,
        env,
    )
