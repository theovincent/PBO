import sys
import argparse
import json
import jax

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train FQI on LQR.")
    addparse(parser, seed=True)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "FQI", "LQR", args.max_bellman_iterations, args.seed)
    p = json.load(open(f"experiments/lqr/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.lqr.utils import define_q, define_data_loader_samples, generate_keys
    from experiments.base.FQI import train

    shuffle_key, _, _ = generate_keys(args.seed)

    data_loader_samples = define_data_loader_samples(
        p["n_discrete_states"] * p["n_discrete_actions"], args.experiment_name, p["batch_size_samples"], shuffle_key
    )
    q = define_q(
        p["n_actions_on_max"],
        p["max_action_on_max"],
        jax.random.PRNGKey(0),
        learning_rate={
            "first": p["starting_lr_fqi"],
            "last": p["ending_lr_fqi"],
            "duration": p["fitting_steps_fqi"] * data_loader_samples.replay_buffer.len // p["batch_size_samples"],
        },
    )

    train("lqr", args, q, p, data_loader_samples)
