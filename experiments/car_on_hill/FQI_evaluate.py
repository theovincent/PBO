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
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Evaluate FQI on Car-On-Hill.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "FQI", "Car-On-Hill", args.bellman_iterations_scope, args.seed)
    p = json.load(open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from pbo.networks.q_architectures import MLPQ
    from pbo.environments.car_on_hill import CarOnHillEnv
    from pbo.utils.params import load_pickled_data

    env = CarOnHillEnv(p["gamma"])
    states_x = np.linspace(-env.max_position, env.max_position, p["n_states_x"])
    states_v = np.linspace(-env.max_velocity, env.max_velocity, p["n_states_v"])

    q = MLPQ(env.state_shape, env.n_actions, p["gamma"], p["layers_dimension"], jax.random.PRNGKey(0))

    iterated_params = load_pickled_data(
        f"experiments/car_on_hill/figures/{args.experiment_name}/FQI/{args.bellman_iterations_scope}_P_{args.seed}"
    )

    def evaluate(
        iteration: int,
        v_list: list,
        q_estimate: list,
        q: MLPQ,
        q_weights: jnp.ndarray,
        horizon: int,
        states_x: np.ndarray,
        states_v: np.ndarray,
    ):
        v_list[iteration] = env.v_mesh(q, q.convert_params.to_params(q_weights), horizon, states_x, states_v)
        q_estimate[iteration] = np.array(
            env.q_estimate_mesh(q, q.convert_params.to_params(q_weights), states_x, states_v)
        )

    manager = multiprocessing.Manager()
    iterated_v = manager.list(
        list(np.nan * np.zeros((args.bellman_iterations_scope + 1, p["n_states_x"], p["n_states_v"])))
    )
    iterated_q_estimate = manager.list(
        list(np.nan * np.zeros((args.bellman_iterations_scope + 1, p["n_states_x"], p["n_states_v"], 2)))
    )

    processes = []
    for iteration in range(args.bellman_iterations_scope + 1):
        processes.append(
            multiprocessing.Process(
                target=evaluate,
                args=(
                    iteration,
                    iterated_v,
                    iterated_q_estimate,
                    q,
                    q.convert_params.to_weights(iterated_params[iteration]),
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
        f"experiments/car_on_hill/figures/{args.experiment_name}/FQI/{args.bellman_iterations_scope}_Q_{args.seed}.npy",
        iterated_q_estimate,
    )
    np.save(
        f"experiments/car_on_hill/figures/{args.experiment_name}/FQI/{args.bellman_iterations_scope}_V_{args.seed}.npy",
        iterated_v,
    )
