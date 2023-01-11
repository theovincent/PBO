import sys
import argparse
import json
import jax

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train FQI on Bicycle.")
    addparse(parser, seed=True)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "FQI", "Bicycle", args.max_bellman_iterations, args.seed)
    p = json.load(open(f"experiments/bicycle/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.bicycle.utils import define_environment, define_q, define_data_loader_samples, generate_keys
    from experiments.base.FQI import train

    shuffle_key, q_key, _ = generate_keys(args.seed)

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])
    data_loader_samples = define_data_loader_samples(
        p["n_samples"], args.experiment_name, p["batch_size_samples"], shuffle_key
    )
    q = define_q(
        env.actions_on_max,
        p["gamma"],
        q_key,
        p["layers_dimension"],
        learning_rate={
            "first": p["starting_lr_fqi"],
            "last": p["ending_lr_fqi"],
            "duration": p["fitting_steps_fqi"] * data_loader_samples.replay_buffer.len // p["batch_size_samples"],
        },
    )

    train("bicycle", args, q, p, data_loader_samples)
