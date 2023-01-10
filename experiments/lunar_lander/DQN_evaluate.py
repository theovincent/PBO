import sys
import argparse
import multiprocessing
import json
import jax
import jax.numpy as jnp
import numpy as np

from experiments.base.parser import addparse


def run_cli(argvs=sys.argv[1:]):
    with jax.default_device(jax.devices("cpu")[0]):
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        parser = argparse.ArgumentParser("Evaluate a DQN on Lunar Lander.")
        addparse(parser, seed=True)
        args = parser.parse_args(argvs)
        print(f"{args.experiment_name}:")
        print(
            f"Evaluating DQN on Lunar Lander with {args.max_bellman_iterations} Bellman iterations and seed {args.seed} ..."
        )
        p = json.load(
            open(f"experiments/lunar_lander/figures/{args.experiment_name}/parameters.json")
        )  # p for parameters

        from experiments.lunar_lander.utils import define_environment, define_q
        from pbo.networks.learnable_q import FullyConnectedQ
        from pbo.utils.params import load_params

        key = jax.random.PRNGKey(args.seed)
        _, q_network_key, _ = jax.random.split(key, 3)

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

        q = define_q(env.actions_on_max, p["gamma"], q_network_key, p["layers_dimension"])
        iterated_params = load_params(
            f"experiments/lunar_lander/figures/{args.experiment_name}/DQN/{args.max_bellman_iterations}_P_{args.seed}"
        )

        def evaluate(iteration: int, j_list: list, q: FullyConnectedQ, q_weights: jnp.ndarray, horizon: int):
            j_list[iteration] = env.evaluate(
                q,
                q.to_params(q_weights),
                horizon,
                p["n_simulations"],
                video_path=f"{args.experiment_name}/DQN/{iteration}_{args.seed}",
            )

        manager = multiprocessing.Manager()
        iterated_j = manager.list(list(np.zeros(args.max_bellman_iterations + 1)))

        processes = []
        for iteration in range(args.max_bellman_iterations + 1):
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(iteration, iterated_j, q, q.to_weights(iterated_params[f"{iteration}"]), p["horizon"]),
                )
            )

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        np.save(
            f"experiments/lunar_lander/figures/{args.experiment_name}/DQN/{args.max_bellman_iterations}_J_{args.seed}.npy",
            iterated_j,
        )
