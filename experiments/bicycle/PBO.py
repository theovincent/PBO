import sys
import argparse
import json
import jax

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train a PBO on Bicycle.")
    addparse(parser, seed=True, architecture=True)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, f"a {args.architecture} PBO", "Bicycle", args.max_bellman_iterations, args.seed)
    p = json.load(open(f"experiments/bicycle/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.bicycle.utils import define_environment, define_q, define_data_loader_samples, generate_keys
    from pbo.weights_collection.weights_buffer import WeightsBuffer
    from pbo.weights_collection.dataloader import WeightsDataLoader
    from experiments.base.PBO_offline import train

    shuffle_key, q_key, pbo_key = generate_keys(args.seed)

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])
    data_loader_samples = define_data_loader_samples(
        p["n_samples"], args.experiment_name, p["batch_size_samples"], shuffle_key
    )
    q = define_q(env.actions_on_max, p["gamma"], q_key, p["layers_dimension"])

    weights_buffer = WeightsBuffer()
    weights_buffer.add(q.to_weights(q.params))

    # Add random weights
    while len(weights_buffer) < p["n_weights"]:
        weights = q.random_init_weights()
        weights_buffer.add(weights)

    weights_buffer.cast_to_jax_array()
    data_loader_weights = WeightsDataLoader(weights_buffer, p["batch_size_weights"], shuffle_key)

    train("bicycle", args, q, p, pbo_key, data_loader_samples, data_loader_weights)
