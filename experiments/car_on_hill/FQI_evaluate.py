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

        parser = argparse.ArgumentParser("Evaluate FQI on Car-On-Hill.")
        addparse(parser, seed=True)
        args = parser.parse_args(argvs)
        print_info(args.experiment_name, "FQI", "Car-On-Hill", args.max_bellman_iterations, args.seed, train=False)
        p = json.load(
            open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json")
        )  # p for parameters

        from experiments.car_on_hill.utils import define_environment, define_q
        from pbo.networks.learnable_q import FullyConnectedQ
        from pbo.utils.params import load_params

        env, states_x, _, states_v, _ = define_environment(p["gamma"], p["n_states_x"], p["n_states_v"])

        q = define_q(env.actions_on_max, p["gamma"], jax.random.PRNGKey(0), p["layers_dimension"])

        iterated_params = load_params(
            f"experiments/car_on_hill/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_P_{args.seed}"
        )

        def evaluate(
            iteration: int,
            v_list: list,
            q_estimate: list,
            q: FullyConnectedQ,
            q_weights: jnp.ndarray,
            horizon: int,
            states_x: np.ndarray,
            states_v: np.ndarray,
        ):
            v_list[iteration] = env.v_mesh(q, q.to_params(q_weights), horizon, states_x, states_v)
            q_estimate[iteration] = np.array(env.q_estimate_mesh(q, q.to_params(q_weights), states_x, states_v))

        manager = multiprocessing.Manager()
        iterated_v = manager.list(
            list(np.nan * np.zeros((args.max_bellman_iterations + 1, p["n_states_x"], p["n_states_v"])))
        )
        iterated_q_estimate = manager.list(
            list(np.nan * np.zeros((args.max_bellman_iterations + 1, p["n_states_x"], p["n_states_v"], 2)))
        )

        processes = []
        for iteration in range(args.max_bellman_iterations + 1):
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(
                        iteration,
                        iterated_v,
                        iterated_q_estimate,
                        q,
                        q.to_weights(iterated_params[f"{iteration}"]),
                        p["horizon"],
                        states_x,
                        states_v,
                    ),
                )
            )

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        np.save(
            f"experiments/car_on_hill/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_Q_{args.seed}.npy",
            iterated_q_estimate,
        )
        np.save(
            f"experiments/car_on_hill/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_V_{args.seed}.npy",
            iterated_v,
        )
