import sys
import argparse
import json
import jax
import numpy as np
from tqdm import tqdm

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    with jax.default_device(jax.devices("cpu")[0]):
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        parser = argparse.ArgumentParser("Evaluate FQI on LQR.")
        addparse(parser, seed=True)
        args = parser.parse_args(argvs)
        print_info(args.experiment_name, "FQI", "LQR", args.max_bellman_iterations, args.seed, train=False)
        p = json.load(open(f"experiments/lqr/figures/{args.experiment_name}/parameters.json"))  # p for parameters

        from experiments.lqr.utils import define_environment, define_q
        from pbo.utils.params import load_params

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["max_discrete_state"])

        q = define_q(p["n_actions_on_max"], p["max_action_on_max"], jax.random.PRNGKey(0))
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
