import sys
import argparse
import multiprocessing
import json
import jax
import numpy as np

from experiments.base.parser import addparse


def run_cli(argvs=sys.argv[1:]):
    with jax.default_device(jax.devices("cpu")[0]):
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        parser = argparse.ArgumentParser("Evaluate a PBO on Bicycle.")
        addparse(parser, seed=True, architecture=True, validation_bellman_iterations=True)
        args = parser.parse_args(argvs)
        print(f"{args.experiment_name}:")
        print(
            f"Evaluating a {args.architecture} PBO on Bicycle with {args.max_bellman_iterations} + {args.validation_bellman_iterations} Bellman iterations and seed {args.seed} ..."
        )
        if args.conv:
            print("PBO with convolutionnal layers.")

        p = json.load(open(f"experiments/bicycle/figures/{args.experiment_name}/parameters.json"))  # p for parameters

        from experiments.bicycle.utils import define_environment, define_q, generate_keys
        from pbo.networks.learnable_q import FullyConnectedQ
        import haiku as hk
        from pbo.networks.learnable_pbo import LinearPBO, DeepPBO
        from pbo.utils.params import load_params

        key = jax.random.PRNGKey(args.seed)
        _, q_key, _ = generate_keys(args.seed)

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

        q = define_q(env.actions_on_max, p["gamma"], q_key, p["layers_dimension"])

        if args.architecture == "linear":
            pbo = LinearPBO(
                q=q,
                max_bellman_iterations=args.max_bellman_iterations,
                add_infinity=True,
                network_key=jax.random.PRNGKey(0),
                learning_rate={"first": 0, "last": 0, "duration": 0},
                initial_weight_std=0.1,
            )
        else:
            pbo = DeepPBO(
                q=q,
                max_bellman_iterations=args.max_bellman_iterations,
                network_key=jax.random.PRNGKey(0),
                layers_dimension=p["pbo_layers_dimension"],
                learning_rate={"first": 0, "last": 0, "duration": 0},
                initial_weight_std=0.1,
                conv=args.conv,
            )
        pbo.params = load_params(
            f"experiments/bicycle/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_P_{args.seed}"
        )

        def evaluate(iteration: int, metrics_list: list, q: FullyConnectedQ, q_params: hk.Params, horizon: int):
            metrics_list[iteration] = env.evaluate(q, q_params, horizon, p["n_simulations"])

        manager = multiprocessing.Manager()
        iterated_metrics = manager.list(
            list(
                np.zeros(
                    (
                        args.max_bellman_iterations + args.validation_bellman_iterations + 1 + int(pbo.add_infinity),
                        p["n_simulations"],
                        2,
                    )
                )
            )
        )

        q_weights = q.to_weights(q.params)

        processes = []
        for iteration in range(args.max_bellman_iterations + args.validation_bellman_iterations + 1):
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(iteration, iterated_metrics, q, q.to_params(q_weights), p["horizon"]),
                )
            )
            q_weights = pbo(pbo.params, q_weights.reshape((1, -1)))[0]

        if pbo.add_infinity:
            q_weights = pbo.fixed_point(pbo.params)
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(-1, iterated_metrics, q, q.to_params(q_weights), p["horizon"]),
                )
            )

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        np.save(
            f"experiments/bicycle/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_M_{args.seed}.npy",
            iterated_metrics,
        )
