import sys
import argparse
import multiprocessing
import json
import jax
from flax.core import FrozenDict
import numpy as np

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Evaluate a PBO on Car-On-Hill.")
    addparse(parser)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "ProFQI", "Car-On-Hill", args.bellman_iterations_scope, args.seed)

    p = json.load(open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from pbo.networks.q_architectures import MLPQ
    from pbo.networks.pbo_architectures import MLPPBO
    from pbo.environments.car_on_hill import CarOnHillEnv
    from pbo.utils.params import load_pickled_data

    env = CarOnHillEnv(p["gamma"])
    states_x = np.linspace(-env.max_position, env.max_position, p["n_states_x"])
    states_v = np.linspace(-env.max_velocity, env.max_velocity, p["n_states_v"])

    q = MLPQ(env.state_shape, env.n_actions, p["gamma"], p["layers_dimension"], jax.random.PRNGKey(0))

    pbo = MLPPBO(q, args.bellman_iterations_scope, p["profqi_features"], jax.random.PRNGKey(0))

    pbo.params = load_pickled_data(
        f"experiments/car_on_hill/figures/{args.experiment_name}/ProFQI/{args.bellman_iterations_scope}_P_{args.seed}_online_params"
    )

    def evaluate(
        iteration: int,
        v_list: list,
        q_estimate: list,
        q: MLPQ,
        q_params: FrozenDict,
        horizon: int,
        states_x: np.ndarray,
        states_v: np.ndarray,
    ):
        v_list[iteration] = env.v_mesh(q, q_params, horizon, states_x, states_v)
        q_estimate[iteration] = np.array(env.q_estimate_mesh(q, q_params, states_x, states_v))

    manager = multiprocessing.Manager()
    iterated_v = manager.list(
        list(
            np.nan
            * np.zeros(
                (
                    args.bellman_iterations_scope + 10 + 1,
                    p["n_states_x"],
                    p["n_states_v"],
                )
            )
        )
    )
    iterated_q_estimate = manager.list(
        list(
            np.nan
            * np.zeros(
                (
                    args.bellman_iterations_scope + 10 + 1,
                    p["n_states_x"],
                    p["n_states_v"],
                    2,
                )
            )
        )
    )

    q_weights = q.convert_params.to_weights(q.params)

    processes = []
    for iteration in range(args.bellman_iterations_scope + 10 + 1):
        processes.append(
            multiprocessing.Process(
                target=evaluate,
                args=(
                    iteration,
                    iterated_v,
                    iterated_q_estimate,
                    q,
                    q.convert_params.to_params(q_weights),
                    p["horizon"],
                    states_x,
                    states_v,
                ),
            )
        )
        q_weights = pbo.apply(pbo.params, q_weights)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    np.save(
        f"experiments/car_on_hill/figures/{args.experiment_name}/ProFQI/{args.bellman_iterations_scope}_V_{args.seed}.npy",
        iterated_v,
    )
    np.save(
        f"experiments/car_on_hill/figures/{args.experiment_name}/ProFQI/{args.bellman_iterations_scope}_Q_{args.seed}.npy",
        iterated_q_estimate,
    )
