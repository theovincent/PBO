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

        parser = argparse.ArgumentParser("Evaluate a FQI on Bicycle.")
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
        print(
            f"Evaluating FQI on Bicycle with {args.max_bellman_iterations} Bellman iterations and seed {args.seed} ..."
        )
        p = json.load(open(f"experiments/bicycle/figures/{args.experiment_name}/parameters.json"))  # p for parameters

        from experiments.bicycle.utils import define_environment
        from pbo.networks.learnable_q import FullyConnectedQ
        from pbo.utils.params import load_params

        key = jax.random.PRNGKey(args.seed)
        _, q_network_key, _ = jax.random.split(key, 3)

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

        q = FullyConnectedQ(
            state_dim=4,
            action_dim=2,
            actions_on_max=env.actions_on_max,
            gamma=p["gamma"],
            network_key=q_network_key,
            layers_dimension=p["layers_dimension"],
            zero_initializer=True,
        )
        iterated_params = load_params(
            f"experiments/bicycle/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_P_{args.seed}"
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
            f"experiments/bicycle/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_M_{args.seed}.npy",
            iterated_metrics,
        )
