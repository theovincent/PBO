import sys
import argparse
import json
import jax

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train DQN on Lunar Lander.")
    addparse(parser, seed=True)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "DQN", "Lunar Lander", args.max_bellman_iterations, args.seed)
    p = json.load(open(f"experiments/lunar_lander/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.lunar_lander.utils import (
        define_environment,
        define_q,
        collect_random_samples,
        collect_samples,
        generate_keys,
    )
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from experiments.base.DQN import train

    sample_key, exploration_key, q_key, _ = generate_keys(args.seed)

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

    replay_buffer = ReplayBuffer(p["max_size"])
    collect_random_samples(env, replay_buffer, p["n_initial_samples"], p["horizon"])

    q = define_q(
        env.actions_on_max,
        p["gamma"],
        q_key,
        p["layers_dimension"],
        learning_rate={
            "first": p["starting_lr_dqn"],
            "last": p["ending_lr_dqn"],
            "duration": args.max_bellman_iterations * p["fitting_updates_dqn"],
        },
    )

    train("lunar_lander", args, q, p, exploration_key, sample_key, replay_buffer, collect_samples, env)
