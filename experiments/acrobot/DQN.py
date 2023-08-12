import sys
import argparse
import json
import jax

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train DQN on Acrobot.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "DQN", "Acrobot", args.bellman_iterations_scope, args.seed)
    p = json.load(open(f"experiments/acrobot/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from pbo.environments.acrobot import AcrobotEnv
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.networks.q_architectures import MLPQ
    from experiments.base.DQN import train

    network_key, key = jax.random.split(jax.random.PRNGKey(args.seed))

    env = AcrobotEnv(jax.random.PRNGKey(p["env_seed"]))
    replay_buffer = ReplayBuffer(p["replay_buffer_size"], p["batch_size"], env.state_shape, lambda x: x)

    q = MLPQ(
        env.state_shape,
        env.n_actions,
        p["gamma"],
        p["features"],
        network_key,
        p["dqn_learning_rate"],
        p["n_training_steps_per_online_update"],
        p["dqn_n_training_steps_per_target_update"],
    )

    train(key, f"experiments/acrobot/figures/{args.experiment_name}/DQN", args, p, q, env, replay_buffer)
