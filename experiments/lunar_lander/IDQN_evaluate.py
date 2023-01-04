import sys
import argparse
import multiprocessing
import json
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np


def run_cli(argvs=sys.argv[1:]):
    with jax.default_device(jax.devices("cpu")[0]):
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        parser = argparse.ArgumentParser("Evaluate a IDQN on Lunar Lander.")
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
            f"Evaluating IDQN on Lunar Lander with {args.max_bellman_iterations} Bellman iterations and seed {args.seed} ..."
        )
        p = json.load(
            open(f"experiments/lunar_lander/figures/{args.experiment_name}/parameters.json")
        )  # p for parameters

        from experiments.lunar_lander.utils import define_environment
        from pbo.networks.learnable_multi_head_q import FullyConnectedMultiHeadQ
        from pbo.utils.params import load_params

        key = jax.random.PRNGKey(args.seed)
        _, q_network_key, _ = jax.random.split(key, 3)

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

        q = FullyConnectedMultiHeadQ(
            n_heads=args.max_bellman_iterations + 1,
            state_dim=8,
            action_dim=1,
            actions_on_max=env.actions_on_max,
            gamma=p["gamma"],
            network_key=q_network_key,
            layers_dimension=p["layers_dimension"],
            zero_initializer=True,
        )
        q.params = load_params(
            f"experiments/lunar_lander/figures/{args.experiment_name}/IDQN/{args.max_bellman_iterations}_P_{args.seed}"
        )

        def evaluate(
            iteration: int,
            j_list: list,
            q: FullyConnectedMultiHeadQ,
            q_weights: jnp.ndarray,
            horizon: int,
        ):
            q_inference = jax.jit(lambda q_params_, state_, action_: q(q_params_, state_, action_)[..., iteration])

            j_list[iteration] = env.evaluate(
                q_inference,
                q.to_params(q_weights),
                horizon,
                p["n_evaluations"],
                video_path=f"{args.experiment_name}/IDQN/{iteration}_{args.seed}",
            )

        manager = multiprocessing.Manager()
        iterated_j = manager.list(list(np.zeros(args.max_bellman_iterations + 1)))

        processes = []
        for iteration in range(args.max_bellman_iterations + 1):
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(iteration, iterated_j, q, q.to_weights(q.params), p["horizon"]),
                )
            )

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        np.save(
            f"experiments/lunar_lander/figures/{args.experiment_name}/IDQN/{args.max_bellman_iterations}_J_{args.seed}.npy",
            iterated_j,
        )
