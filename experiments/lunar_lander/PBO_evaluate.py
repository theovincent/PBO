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

        parser = argparse.ArgumentParser("Evaluate a PBO on Lunar Lander.")
        addparse(parser, seed=True, architecture=True, validation_bellman_iterations=True)
        args = parser.parse_args(argvs)
        print(f"{args.experiment_name}:")
        print(
            f"Evaluating a {args.architecture} PBO on Lunar Lander with {args.max_bellman_iterations} + {args.validation_bellman_iterations} Bellman iterations and seed {args.seed} ..."
        )
        if args.conv:
            print("PBO with convolutionnal layers.")
        p = json.load(
            open(f"experiments/lunar_lander/figures/{args.experiment_name}/parameters.json")
        )  # p for parameters

        from experiments.lunar_lander.utils import define_environment, define_q
        from pbo.networks.learnable_q import FullyConnectedQ
        import haiku as hk
        from pbo.networks.learnable_pbo import LinearPBO, DeepPBO
        from pbo.utils.params import load_params

        key = jax.random.PRNGKey(args.seed)
        _, q_network_key, pbo_network_key = jax.random.split(key, 3)

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

        q = define_q(env.actions_on_max, p["gamma"], q_network_key, p["layers_dimension"])

        if args.architecture == "linear":
            add_infinity = True
            pbo = LinearPBO(
                q=q,
                max_bellman_iterations=args.max_bellman_iterations,
                add_infinity=add_infinity,
                network_key=pbo_network_key,
                learning_rate={"first": 0, "last": 0, "duration": 0},
                initial_weight_std=0.1,
            )
        else:
            add_infinity = False
            pbo = DeepPBO(
                q=q,
                max_bellman_iterations=args.max_bellman_iterations,
                network_key=pbo_network_key,
                layers_dimension=p["pbo_layers_dimension"],
                learning_rate={"first": 0, "last": 0, "duration": 0},
                initial_weight_std=0.1,
                conv=args.conv,
            )
        pbo.params = load_params(
            f"experiments/lunar_lander/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_P_{args.seed}"
        )

        def evaluate(
            iteration: int,
            j_list: list,
            q: FullyConnectedQ,
            q_params: hk.Params,
            horizon: int,
        ):
            j_list[iteration] = env.evaluate(
                q,
                q_params,
                horizon,
                p["n_simulations"],
                video_path=f"{args.experiment_name}/PBO_{args.architecture}/{iteration}_{args.seed}",
            )

        manager = multiprocessing.Manager()
        iterated_j = manager.list(
            list(np.zeros(args.max_bellman_iterations + args.validation_bellman_iterations + 1 + int(add_infinity)))
        )

        q_weights = q.to_weights(q.params)

        processes = []
        for iteration in range(args.max_bellman_iterations + args.validation_bellman_iterations + 1):
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(iteration, iterated_j, q, q.to_params(q_weights), p["horizon"]),
                )
            )
            q_weights = pbo(pbo.params, q_weights.reshape((1, -1)))[0]

        if add_infinity:
            q_weights = pbo.fixed_point(pbo.params)
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(-1, iterated_j, q, q.to_params(q_weights), p["horizon"]),
                )
            )

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        np.save(
            f"experiments/lunar_lander/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_J_{args.seed}.npy",
            iterated_j,
        )
