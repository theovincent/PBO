import sys
import argparse
import json
import jax
import numpy as np

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train ProDQN on Atari.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "ProDQN", "Atari", args.bellman_iterations_scope, args.seed)
    p = json.load(
        open(f"experiments/atari/figures/{args.experiment_name.split('/')[0]}/parameters.json")
    )  # p for parameters

    from pbo.environments.atari import AtariEnv
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.networks.q_architectures import MLPQ
    from pbo.networks.pbo_architectures import SplittedMLPPBO
    from experiments.base.ProDQN import train

    network_key, key = jax.random.split(jax.random.PRNGKey(args.seed))

    env = AtariEnv(args.experiment_name.split("/")[1])
    replay_buffer = ReplayBuffer(
        p["replay_buffer_size"], p["batch_size"], env.state_shape, np.float32, lambda x: np.clip(x, -1, 1)
    )

    q = MLPQ(env.state_shape, env.n_actions, p["gamma"], [], network_key)

    pbo = SplittedMLPPBO(
        q,
        args.bellman_iterations_scope,
        1000,
        p["prodqn_features"],
        jax.random.split(network_key)[0],
        p["prodqn_learning_rate"],
        p["prodqn_optimizer_eps"],
        p["n_training_steps_per_online_update"],
        p["prodqn_n_training_steps_per_target_update"],
        p["prodqn_n_current_weights"],
        p["prodqn_n_training_steps_per_current_weights_update"],
    )

    train(key, f"experiments/atari/figures/{args.experiment_name}/ProDQN", args, p, pbo, env, replay_buffer)
