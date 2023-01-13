import sys
import argparse
import multiprocessing
import json
import jax
import jax.numpy as jnp
import numpy as np

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    with jax.default_device(jax.devices("cpu")[0]):
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        parser = argparse.ArgumentParser("Evaluate a FQI on Bicycle.")
        addparse(parser, seed=True)
        args = parser.parse_args(argvs)
        print_info(args.experiment_name, "FQI", "Bicycle", args.max_bellman_iterations, args.seed, train=False)
        p = json.load(
            open(f"experiments/bicycle_offline/figures/{args.experiment_name}/parameters.json")
        )  # p for parameters

        from experiments.bicycle_offline.utils import define_environment, define_q
        from pbo.networks.learnable_q import FullyConnectedQ
        from pbo.utils.params import load_params

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

        q = define_q(env.actions_on_max, p["gamma"], jax.random.PRNGKey(0), p["layers_dimension"])
        iterated_params = load_params(
            f"experiments/bicycle_offline/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_P_{args.seed}"
        )

        def evaluate(iteration: int, metrics_list: list, q: FullyConnectedQ, q_weights: jnp.ndarray, horizon: int):
            metrics_list[iteration] = env.evaluate(q, q.to_params(q_weights), horizon, p["n_simulations"])

        manager = multiprocessing.Manager()
        iterated_metrics = manager.list(list(np.zeros((args.max_bellman_iterations + 1, p["n_simulations"], 2))))

        processes = []
        for iteration in range(args.max_bellman_iterations + 1):
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(iteration, iterated_metrics, q, q.to_weights(iterated_params[f"{iteration}"]), p["horizon"]),
                )
            )

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        np.save(
            f"experiments/bicycle_offline/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_M_{args.seed}.npy",
            iterated_metrics,
        )
