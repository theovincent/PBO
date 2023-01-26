import sys
import argparse
import json
import jax

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train a PBO on LQR.")
    addparse(parser, seed=True, architecture=True, validation_bellman_iterations=True)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, f"a {args.architecture} PBO", "LQR", args.max_bellman_iterations, args.seed)
    p = json.load(open(f"experiments/lqr/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.lqr.utils import define_environment, define_data_loader_samples, define_q, generate_keys
    from pbo.weights_collection.weights_buffer import WeightsBuffer
    from pbo.weights_collection.dataloader import WeightsDataLoader
    from experiments.base.PBO_offline import train

    shuffle_key, q_key, pbo_key = generate_keys(args.seed)

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["max_discrete_state"])
    data_loader_samples = define_data_loader_samples(
        p["n_discrete_states"] * p["n_discrete_actions"], args.experiment_name, p["batch_size_samples"], shuffle_key
    )
    q = define_q(
        p["n_actions_on_max"],
        p["max_action_on_max"],
        env.optimal_weights[2] if p["q_dim"] == 2 else None,
        jax.random.PRNGKey(0),
    )

    weights_buffer = WeightsBuffer()
    weights_buffer.add(q.to_weights(q.params))

    q_random = define_q(
        p["n_actions_on_max"],
        p["max_action_on_max"],
        env.optimal_weights[2] if p["q_dim"] == 2 else None,
        q_key,
        zero_initializer=False,
    )
    # Add random weights
    while len(weights_buffer) < p["n_weights"]:
        weights = q_random.random_init_weights()
        weights_buffer.add(weights)

    weights_buffer.cast_to_jax_array()
    data_loader_weights = WeightsDataLoader(weights_buffer, p["batch_size_weights"], shuffle_key)

    train("lqr", args, q, p, pbo_key, data_loader_samples, data_loader_weights)
