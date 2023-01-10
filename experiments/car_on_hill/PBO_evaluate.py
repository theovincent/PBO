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

        parser = argparse.ArgumentParser("Evaluate a PBO on Car-On-Hill.")
        addparse(parser, seed=True, architecture=True, validation_bellman_iterations=True)
        args = parser.parse_args(argvs)
        print(f"{args.experiment_name}:")
        print(
            f"Evaluating a {args.architecture} PBO on Car-On-Hill with {args.max_bellman_iterations} + {args.validation_bellman_iterations} Bellman iterations and seed {args.seed} ..."
        )
        if args.conv:
            print("PBO with convolutionnal layers.")

        p = json.load(
            open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json")
        )  # p for parameters

        from experiments.car_on_hill.utils import define_environment, define_q
        from pbo.networks.learnable_q import FullyConnectedQ
        import haiku as hk
        from pbo.networks.learnable_pbo import LinearPBO, DeepPBO
        from pbo.utils.params import load_params

        key = jax.random.PRNGKey(args.seed)
        _, q_network_key, pbo_network_key = jax.random.split(key, 3)

        env, states_x, _, states_v, _ = define_environment(p["gamma"], p["n_states_x"], p["n_states_v"])

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
            f"experiments/car_on_hill/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_P_{args.seed}"
        )

        def evaluate(
            iteration: int,
            v_list: list,
            q_estimate: list,
            q: FullyConnectedQ,
            q_params: hk.Params,
            horizon: int,
            states_x: np.ndarray,
            states_v: np.ndarray,
        ):
            v_list[iteration] = env.v_mesh(q, q_params, horizon, states_x, states_v)
            q_estimate[iteration] = np.array(env.q_estimate_mesh(q, q_params, states_x, states_v))

        manager = multiprocessing.Manager()
        iterated_v = manager.list(
            list(
                np.zeros(
                    (
                        args.max_bellman_iterations + args.validation_bellman_iterations + 1 + int(add_infinity),
                        p["n_states_x"],
                        p["n_states_v"],
                    )
                )
            )
        )
        iterated_q_estimate = manager.list(
            list(
                np.zeros(
                    (
                        args.max_bellman_iterations + args.validation_bellman_iterations + 1 + int(add_infinity),
                        p["n_states_x"],
                        p["n_states_v"],
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
                    args=(
                        iteration,
                        iterated_v,
                        iterated_q_estimate,
                        q,
                        q.to_params(q_weights),
                        p["horizon"],
                        states_x,
                        states_v,
                    ),
                )
            )
            q_weights = pbo(pbo.params, q_weights.reshape((1, -1)))[0]

        if add_infinity:
            q_weights = pbo.fixed_point(pbo.params)
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(
                        -1,
                        iterated_v,
                        iterated_q_estimate,
                        q,
                        q.to_params(q_weights),
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
            f"experiments/car_on_hill/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_V_{args.seed}.npy",
            iterated_v,
        )
        np.save(
            f"experiments/car_on_hill/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_Q_{args.seed}.npy",
            iterated_q_estimate,
        )
