import sys
import argparse
import json
import jax

from experiments.base.parser import addparse


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train IFQI on Car-On-Hill.")
    addparse(parser, seed=True)
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    print(f"Training IFQI on Car-On-Hill with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}...")
    p = json.load(open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.car_on_hill.utils import (
        define_environment,
        define_q_multi_head,
        define_data_loader_samples,
        generate_keys,
    )
    from experiments.base.IFQI import train

    shuffle_key, q_key, _ = generate_keys(args.seed)

    env, _, _, _, _ = define_environment(p["gamma"], p["n_states_x"], p["n_states_v"])
    data_loader_samples = define_data_loader_samples(
        p["n_random_samples"] + p["n_oriented_samples"], args.experiment_name, p["batch_size_samples"], shuffle_key
    )
    q = define_q_multi_head(
        args.max_bellman_iterations + 1,
        env.actions_on_max,
        p["gamma"],
        q_key,
        p["layers_dimension"],
        learning_rate={
            "first": p["starting_lr_ifqi"],
            "last": p["ending_lr_ifqi"],
            "duration": p["training_steps_ifqi"]
            * p["fitting_steps_ifqi"]
            * data_loader_samples.replay_buffer.len
            // p["batch_size_samples"],
        },
    )

    train("car_on_hill", args, q, p, data_loader_samples)
