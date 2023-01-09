import sys
import argparse
import json
import jax
import numpy as np
from tqdm import tqdm


def run_cli(argvs=sys.argv[1:]):
    with jax.default_device(jax.devices("cpu")[0]):
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        parser = argparse.ArgumentParser("Evaluate FQI on LQR.")
        parser.add_argument(
            "-e",
            "--experiment_name",
            help="Experiment name.",
            type=str,
            required=True,
        )
        parser.add_argument(
            "-s",
            "--seed",
            help="Seed of the training.",
            type=int,
            required=True,
        )
        parser.add_argument(
            "-b",
            "--max_bellman_iterations",
            help="Maximum number of Bellman iteration.",
            type=int,
            required=True,
        )
        args = parser.parse_args(argvs)
        print(f"{args.experiment_name}:")
        print(f"Evaluating FQI on LQR with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}...")
        p = json.load(open(f"experiments/lqr/figures/{args.experiment_name}/parameters.json"))  # p for parameters

        from experiments.lqr.utils import define_environment
        from pbo.networks.learnable_q import LQRQ
        from pbo.utils.params import load_params

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["max_discrete_state"])

        q = LQRQ(
            n_actions_on_max=p["n_actions_on_max"],
            max_action_on_max=p["max_action_on_max"],
            network_key=jax.random.PRNGKey(0),
            zero_initializer=True,
        )
        iterated_params = load_params(
            f"experiments/lqr/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_P_{args.seed}"
        )

        weights = np.zeros((args.max_bellman_iterations + 1, q.weights_dimension))

        for iteration in tqdm(range(args.max_bellman_iterations + 1)):
            weights[iteration] = q.to_weights(iterated_params[f"{iteration}"])

        np.save(
            f"experiments/lqr/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_W_{args.seed}.npy",
            weights,
        )
        np.save(
            f"experiments/lqr/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_V_{args.seed}.npy",
            env.greedy_V(weights),
        )
