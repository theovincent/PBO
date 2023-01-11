import sys
import argparse
import json
import jax

from experiments.base.parser import addparse


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train FQI on Chain Walk.")
    addparse(parser, seed=True)
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    print(f"Training FQI on Chain Walk with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}...")
    p = json.load(open(f"experiments/chain_walk/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.chain_walk.utils import define_environment, define_q, define_data_loader_samples, generate_keys
    from experiments.base.FQI import train

    shuffle_key, _, _ = generate_keys(args.seed)

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["n_states"], p["sucess_probability"], p["gamma"])
    data_loader_samples = define_data_loader_samples(
        p["n_states"] * env.n_actions * p["n_repetitions"], args.experiment_name, p["batch_size_samples"], shuffle_key
    )
    q = define_q(
        p["n_states"],
        env.n_actions,
        p["gamma"],
        jax.random.PRNGKey(0),
        learning_rate={
            "first": p["starting_lr_fqi"],
            "last": p["ending_lr_fqi"],
            "duration": p["fitting_steps_fqi"] * data_loader_samples.replay_buffer.len // p["batch_size_samples"],
        },
    )

    train("chain_walk", args, q, p, data_loader_samples)
