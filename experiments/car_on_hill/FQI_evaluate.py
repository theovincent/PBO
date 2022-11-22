import sys
import argparse
import multiprocessing
import json
import jax
import jax.numpy as jnp
import numpy as np


def run_cli(argvs=sys.argv[1:]):
    with jax.default_device(jax.devices("cpu")[0]):
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        parser = argparse.ArgumentParser("Evaluate a FQI on Car-On-Hill.")
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
        print(
            f"Evaluating FQI on Car-On-Hill with {args.max_bellman_iterations} Bellman iterations and seed {args.seed} ..."
        )
        p = json.load(open("experiments/car_on_hill/parameters.json"))  # p for parameters

        from experiments.car_on_hill.utils import define_environment
        from pbo.networks.learnable_q import FullyConnectedQ
        from pbo.utils.params import load_params

        key = jax.random.PRNGKey(args.seed)
        _, q_network_key, _ = jax.random.split(key, 3)

        env, states_x, _, states_v, _ = define_environment(p["gamma"], p["n_states_x"], p["n_states_v"])

        q = FullyConnectedQ(
            state_dim=2,
            action_dim=1,
            actions_on_max=env.actions_on_max,
            gamma=p["gamma"],
            network_key=q_network_key,
            layers_dimension=p["layers_dimension"],
            zero_initializer=True,
        )
        iterated_params = load_params(
            f"experiments/car_on_hill/figures/data/FQI/{args.max_bellman_iterations}_P_{args.seed}"
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
        iterated_v = manager.list(list(np.zeros((args.max_bellman_iterations + 1, p["n_states_x"], p["n_states_v"]))))
        iterated_q_estimate = manager.list(
            list(np.zeros((args.max_bellman_iterations + 1, p["n_states_x"], p["n_states_v"], 2)))
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
            f"experiments/car_on_hill/figures/data/FQI/{args.max_bellman_iterations}_Q_{args.seed}.npy",
            iterated_q_estimate,
        )
        np.save(f"experiments/car_on_hill/figures/data/FQI/{args.max_bellman_iterations}_V_{args.seed}.npy", iterated_v)
