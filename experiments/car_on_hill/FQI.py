import sys
import argparse
import json
import jax

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train FQI on Car-On-Hill.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "FQI", "Car-On-Hill", args.bellman_iterations_scope, args.seed)
    p = json.load(open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from pbo.environments.car_on_hill import CarOnHillEnv
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.networks.q_architectures import MLPQ
    from experiments.base.FQI import train

    network_key, key = jax.random.split(jax.random.PRNGKey(args.seed))

    replay_buffer = ReplayBuffer(
        p["n_random_samples"] + p["n_oriented_samples"], p["batch_size_samples"], (2,), float, lambda x: x
    )
    replay_buffer.load(f"experiments/car_on_hill/figures/{args.experiment_name}/replay_buffer")

    env = CarOnHillEnv(p["gamma"])

    q = MLPQ(
        env.state_shape,
        env.n_actions,
        p["gamma"],
        p["layers_dimension"],
        network_key,
        p["fqi_learning_rate"],
        p["fqi_optimizer_eps"],
        n_training_steps_per_online_update=1,  # always wanted when called
        n_training_steps_per_target_update=1,  # always wanted when called
    )

    train(key, f"experiments/car_on_hill/figures/{args.experiment_name}/FQI", args, p, q, replay_buffer)
