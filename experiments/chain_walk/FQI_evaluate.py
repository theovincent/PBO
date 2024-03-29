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

        parser = argparse.ArgumentParser("Evaluate FQI on Chain Walk.")
        addparse(parser, seed=True)
        args = parser.parse_args(argvs)
        print_info(args.experiment_name, "FQI", "Chain Walk", args.max_bellman_iterations, args.seed, train=False)
        p = json.load(
            open(f"experiments/chain_walk/figures/{args.experiment_name}/parameters.json")
        )  # p for parameters

        from experiments.chain_walk.utils import define_environment, define_q
        from pbo.utils.params import load_params

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["n_states"], p["sucess_probability"], p["gamma"])

        q = define_q(p["n_states"], env.n_actions, p["gamma"], jax.random.PRNGKey(0))
        iterated_params = load_params(
            f"experiments/chain_walk/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_P_{args.seed}"
        )

        q_functions = np.nan * np.zeros((args.max_bellman_iterations + 1, env.n_states, env.n_actions))
        v_functions = np.nan * np.zeros((args.max_bellman_iterations + 1, env.n_states))

        for iteration in tqdm(range(args.max_bellman_iterations + 1)):
            q_functions[iteration] = env.discretize(q, iterated_params[f"{iteration}"])
            policy_q = q_functions[iteration].argmax(axis=1)
            v_functions[iteration] = env.value_function(policy_q)

        np.save(
            f"experiments/chain_walk/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_Q_{args.seed}.npy",
            q_functions,
        )
        np.save(
            f"experiments/chain_walk/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_V_{args.seed}.npy",
            v_functions,
        )
