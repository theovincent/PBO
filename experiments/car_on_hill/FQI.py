import sys
import argparse
import json
import jax

from experiments.base.parser import addparse


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train FQI on Car-On-Hill.")
    addparse(parser, seed=True)
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    print(f"Training FQI on Car-On-Hill with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}...")
    p = json.load(open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.car_on_hill.utils import define_environment, define_q, define_data_loader_samples
    from experiments.base.FQI import train

    key = jax.random.PRNGKey(args.seed)
    shuffle_key, q_network_key, _ = jax.random.split(
        key, 3
    )  # 3 keys are generated to be coherent with the other trainings

    env, _, _, _, _ = define_environment(p["gamma"], p["n_states_x"], p["n_states_v"])
    data_loader_samples = define_data_loader_samples(
        p["n_random_samples"] + p["n_oriented_samples"], args.experiment_name, p["batch_size_samples"], shuffle_key
    )
    q = define_q(
        env.actions_on_max,
        p["gamma"],
        q_network_key,
        p["layers_dimension"],
        learning_rate={
            "first": p["starting_lr_fqi"],
            "last": p["ending_lr_fqi"],
            "duration": p["fitting_steps_fqi"] * data_loader_samples.replay_buffer.len // p["batch_size_samples"],
        },
    )

    train("car_on_hill", args, q, p, data_loader_samples)
