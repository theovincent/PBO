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

        parser = argparse.ArgumentParser("Evaluate a PBO on Chain Walk.")
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
            choices=["linear", "max_linear", "deep"],
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
            f"Evaluating a {args.architecture} PBO on Chain Walk with {args.max_bellman_iterations} + {args.validation_bellman_iterations} Bellman iterations and seed {args.seed} ..."
        )
        p = json.load(
            open(f"experiments/chain_walk/figures/{args.experiment_name}/parameters.json")
        )  # p for parameters

        from experiments.chain_walk.utils import define_environment
        from pbo.networks.learnable_q import TableQ
        import haiku as hk
        from pbo.networks.learnable_pbo import LinearPBO, MaxLinearPBO, DeepPBO
        from pbo.utils.params import load_params

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["n_states"], p["sucess_probability"], p["gamma"])

        q = TableQ(
            n_states=p["n_states"],
            n_actions=env.n_actions,
            gamma=p["gamma"],
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
        elif args.architecture == "max_linear":
            add_infinity = False
            pbo = MaxLinearPBO(
                q=q,
                max_bellman_iterations=args.max_bellman_iterations,
                network_key=jax.random.PRNGKey(0),
                learning_rate={"first": 0, "last": 0, "duration": 0},
                n_actions=env.n_actions,
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
            f"experiments/chain_walk/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_P_{args.seed}"
        )

        iterated_q_estimate = np.zeros(
            (
                args.max_bellman_iterations + args.validation_bellman_iterations + int(add_infinity) + 1,
                env.n_states,
                env.n_actions,
            )
        )
        iterated_v = np.zeros(
            (args.max_bellman_iterations + args.validation_bellman_iterations + int(add_infinity) + 1, env.n_states)
        )
        q_weights = q.to_weights(q.params)

        for iteration in tqdm(range(args.max_bellman_iterations + args.validation_bellman_iterations + 1)):
            iterated_q_estimate[iteration] = env.discretize(q, q.to_params(q_weights))
            policy_q = iterated_q_estimate[iteration].argmax(axis=1)
            iterated_v[iteration] = env.value_function(policy_q)

            q_weights = pbo(pbo.params, q_weights.reshape((1, -1)))[0]

        if add_infinity:
            q_weights = pbo.fixed_point(pbo.params)

            iterated_q_estimate[-1] = env.discretize(q, q.to_params(q_weights))
            policy_q = iterated_q_estimate[-1].argmax(axis=1)
            iterated_v[-1] = env.value_function(policy_q)

        np.save(
            f"experiments/chain_walk/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_Q_{args.seed}.npy",
            iterated_q_estimate,
        )
        np.save(
            f"experiments/chain_walk/figures/{args.experiment_name}/PBO_{args.architecture}/{args.max_bellman_iterations}_V_{args.seed}.npy",
            iterated_v,
        )
