import sys
import argparse
import json
import jax

from experiments.base.parser import addparse


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train IFQI on Bicycle.")
    addparse(parser, seed=True)
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    print(f"Training IFQI on Bicycle with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}...")
    p = json.load(open(f"experiments/bicycle/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.bicycle.utils import define_environment, define_q_multi_head, define_data_loader_samples
    from experiments.base.IFQI import train

    key = jax.random.PRNGKey(args.seed)
    shuffle_key, q_network_key, _ = jax.random.split(
        key, 3
    )  # 3 keys are generated to be coherent with the other trainings

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])
    data_loader_samples = define_data_loader_samples(
        p["n_samples"], args.experiment_name, p["batch_size_samples"], shuffle_key
    )
    q = define_q_multi_head(
        args.max_bellman_iterations + 1,
        env.actions_on_max,
        p["gamma"],
        q_network_key,
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

    train("bicycle", args, q, p, data_loader_samples)
