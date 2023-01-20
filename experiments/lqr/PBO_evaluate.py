import sys
import argparse
import json
import jax
import numpy as np
from tqdm import tqdm

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    with jax.default_device(jax.devices("cpu")[0]):
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        parser = argparse.ArgumentParser("Evaluate a PBO on LQR.")
        addparse(parser, seed=True, architecture=True, validation_bellman_iterations=True)
        args = parser.parse_args(argvs)
        print_info(
            args.experiment_name,
            f"a {args.architecture} PBO",
            "LQR",
            args.max_bellman_iterations,
            args.seed,
            train=False,
        )
        p = json.load(open(f"experiments/lqr/figures/{args.experiment_name}/parameters.json"))  # p for parameters

        from experiments.lqr.utils import define_environment, define_q
        from pbo.networks.learnable_pbo import LinearPBO, CustomLinearPBO, DeepPBO
        from pbo.utils.params import load_params

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["max_discrete_state"])
        q = define_q(
            p["n_actions_on_max"],
            p["max_action_on_max"],
            env.optimal_weights[2] if p["q_dim"] == 2 else None,
            jax.random.PRNGKey(0),
        )

        if args.architecture == "linear":
            pbo = LinearPBO(
                q=q,
                max_bellman_iterations=args.max_bellman_iterations,
                add_infinity=True,
                network_key=jax.random.PRNGKey(0),
                learning_rate={"first": 0, "last": 0, "duration": 0},
                initial_weight_std=0.1,
            )
        elif args.architecture == "custom_linear":
            pbo = CustomLinearPBO(
                q=q,
                max_bellman_iterations=args.max_bellman_iterations,
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
                conv=False,
            )
        pbo.params = load_params(
            f"experiments/lqr/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_P_{args.seed}"
        )

        weights = np.zeros(
            (
                args.max_bellman_iterations + args.validation_bellman_iterations + int(pbo.add_infinity) + 1,
                q.weights_dimension,
            )
        )

        q_weights = q.to_weights(q.params)

        for iteration in tqdm(range(args.max_bellman_iterations + args.validation_bellman_iterations + 1)):
            weights[iteration] = q_weights
            q_weights = pbo(pbo.params, q_weights.reshape((1, -1)))[0]

        if pbo.add_infinity:
            weights[-1] = pbo.fixed_point(pbo.params)

        np.save(
            f"experiments/lqr/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_W_{args.seed}.npy",
            weights,
        )
        np.save(
            f"experiments/lqr/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_V_{args.seed}.npy",
            env.greedy_V(weights),
        )
