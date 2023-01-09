import sys
import argparse
import json
import jax
import numpy as np
from tqdm import tqdm


def run_cli(argvs=sys.argv[1:]):
    with jax.default_device(jax.devices("cpu")[0]):
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        parser = argparse.ArgumentParser("Evaluate a PBO on LQR.")
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
        parser.add_argument(
            "-a",
            "--architecture",
            help="Class of the PBO.",
            choices=["linear", "custom_linear", "deep"],
            required=True,
        )
        parser.add_argument(
            "-v",
            "--validation_bellman_iterations",
            help="Number of Bellman iteration to validate on.",
            default=10,
            type=int,
        )
        args = parser.parse_args(argvs)
        print(f"{args.experiment_name}:")
        print(
            f"Evaluating a {args.architecture} PBO on LQR with {args.max_bellman_iterations} + {args.validation_bellman_iterations} Bellman iterations and seed {args.seed} ..."
        )
        p = json.load(open(f"experiments/lqr/figures/{args.experiment_name}/parameters.json"))  # p for parameters

        from experiments.lqr.utils import define_environment
        from pbo.networks.learnable_q import LQRQ
        from pbo.networks.learnable_pbo import LinearPBO, CustomLinearPBO, DeepPBO
        from pbo.utils.params import load_params

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["max_discrete_state"])

        q = LQRQ(
            n_actions_on_max=p["n_actions_on_max"],
            max_action_on_max=p["max_action_on_max"],
            network_key=jax.random.PRNGKey(0),
            zero_initializer=True,
        )

        if args.architecture == "linear":
            add_infinity = True
            pbo = LinearPBO(
                q=q,
                max_bellman_iterations=args.max_bellman_iterations,
                add_infinity=add_infinity,
                network_key=jax.random.PRNGKey(0),
                learning_rate={"first": 0, "last": 0, "duration": 0},
                initial_weight_std=0.1,
            )
        elif args.architecture == "custom_linear":
            add_infinity = False
            pbo = CustomLinearPBO(
                q=q,
                max_bellman_iterations=args.max_bellman_iterations,
                network_key=jax.random.PRNGKey(0),
                learning_rate={"first": 0, "last": 0, "duration": 0},
                initial_weight_std=0.1,
            )
        else:
            add_infinity = False
            pbo = DeepPBO(
                q=q,
                max_bellman_iterations=args.max_bellman_iterations,
                network_key=jax.random.PRNGKey(0),
                layers_dimension=p["pbo_layers_dimension"],
                learning_rate={"first": 0, "last": 0, "duration": 0},
                initial_weight_std=0.1,
                conv=False,
            )
        pbo.params = load_params(
            f"experiments/lqr/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_P_{args.seed}"
        )

        weights = np.zeros(
            (
                args.max_bellman_iterations + args.validation_bellman_iterations + int(add_infinity) + 1,
                q.weights_dimension,
            )
        )

        q_weights = q.to_weights(q.params)

        for iteration in tqdm(range(args.max_bellman_iterations + args.validation_bellman_iterations + 1)):
            weights[iteration] = q_weights
            q_weights = pbo(pbo.params, q_weights.reshape((1, -1)))[0]

        if add_infinity:
            weights[-1] = pbo.fixed_point(pbo.params)

        np.save(
            f"experiments/lqr/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_W_{args.seed}.npy",
            weights,
        )
        np.save(
            f"experiments/lqr/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_V_{args.seed}.npy",
            env.greedy_V(weights),
        )
