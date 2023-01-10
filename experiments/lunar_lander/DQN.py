import sys
import argparse
import json
import jax

from experiments.base.parser import addparse


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train DQN on Lunar Lander.")
    addparse(parser, seed=True)
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    print(f"Training DQN on Lunar Lander with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}...")
    p = json.load(open(f"experiments/lunar_lander/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.lunar_lander.utils import define_environment, define_q, collect_random_samples, collect_samples
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from experiments.base.DQN import train

    key = jax.random.PRNGKey(args.seed)
    sample_key, q_network_key, _ = jax.random.split(
        key, 3
    )  # 3 keys are generated to be coherent with the other trainings

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

    replay_buffer = ReplayBuffer(p["max_size"])
    collect_random_samples(env, replay_buffer, p["n_initial_samples"], p["horizon"])

    q = define_q(
        env.actions_on_max,
        p["gamma"],
        q_network_key,
        p["layers_dimension"],
        learning_rate={
            "first": p["starting_lr_dqn"],
            "last": p["ending_lr_dqn"],
            "duration": args.max_bellman_iterations * p["fitting_updates_dqn"],
        },
    )

    train("lunar_lander", args, q, p, replay_buffer, collect_samples, sample_key, env)
