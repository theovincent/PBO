import sys
import argparse
import json
import jax

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train IDQN on Bicycle.")
    addparse(parser, seed=True)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "IDQN", "Bicycle", args.max_bellman_iterations, args.seed)
    p = json.load(
        open(f"experiments/bicycle_online/figures/{args.experiment_name}/parameters.json")
    )  # p for parameters

    from experiments.bicycle_online.utils import (
        define_environment,
        define_q_multi_head,
        collect_random_samples,
        collect_samples_multi_head,
        generate_keys,
    )
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from experiments.base.IDQN import train

    sample_key, exploration_key, q_key, _ = generate_keys(args.seed)

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])
    replay_buffer = ReplayBuffer(p["max_size"])
    collect_random_samples(env, sample_key, replay_buffer, p["n_initial_samples"], 20)
    q = define_q_multi_head(
        args.max_bellman_iterations + 1,
        env.actions_on_max,
        p["gamma"],
        q_key,
        p["layers_dimension"],
        learning_rate={
            "first": p["starting_lr_idqn"],
            "last": p["ending_lr_idqn"],
            "duration": p["training_steps_idqn"] * p["fitting_steps_idqn"],
        },
    )

    train("bicycle_online", args, q, p, exploration_key, sample_key, replay_buffer, collect_samples_multi_head, env)
