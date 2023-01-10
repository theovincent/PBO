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

        parser = argparse.ArgumentParser("Evaluate a IFQI on Bicycle.")
        addparse(parser, seed=True)
        args = parser.parse_args(argvs)
        print(f"{args.experiment_name}:")
        print(
            f"Evaluating IFQI on Bicycle with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}..."
        )
        p = json.load(open(f"experiments/bicycle/figures/{args.experiment_name}/parameters.json"))  # p for parameters

        from experiments.bicycle.utils import define_environment, define_q_multi_head
        from pbo.networks.learnable_multi_head_q import FullyConnectedMultiHeadQ
        from pbo.utils.params import load_params

        key = jax.random.PRNGKey(args.seed)
        _, q_network_key, _ = jax.random.split(key, 3)

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

        q = define_q_multi_head(
            args.max_bellman_iterations + 1, env.actions_on_max, p["gamma"], q_network_key, p["layers_dimension"]
        )
        q.params = load_params(
            f"experiments/bicycle/figures/{args.experiment_name}/IFQI/{args.max_bellman_iterations}_P_{args.seed}"
        )

        def evaluate(
            iteration: int, metrics_list: list, q: FullyConnectedMultiHeadQ, q_weights: jnp.ndarray, horizon: int
        ):
            q_inference = jax.jit(lambda q_params_, state_, action_: q(q_params_, state_, action_)[..., iteration])

            metrics_list[iteration] = env.evaluate(q_inference, q.to_params(q_weights), horizon, p["n_simulations"])

        manager = multiprocessing.Manager()
        iterated_metrics = manager.list(list(np.zeros((args.max_bellman_iterations + 1, p["n_simulations"], 2))))

        processes = []
        for iteration in range(args.max_bellman_iterations + 1):
            processes.append(
                multiprocessing.Process(
                    target=evaluate, args=(iteration, iterated_metrics, q, q.to_weights(q.params), p["horizon"])
                )
            )

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        np.save(
            f"experiments/bicycle/figures/{args.experiment_name}/IFQI/{args.max_bellman_iterations}_M_{args.seed}.npy",
            iterated_metrics,
        )
