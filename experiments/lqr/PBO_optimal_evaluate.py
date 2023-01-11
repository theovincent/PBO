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

        parser = argparse.ArgumentParser("Compute PBO optimal on LQR.")
        addparse(parser, validation_bellman_iterations=True)
        args = parser.parse_args(argvs)
        print_info(args.experiment_name, "an optimal PBO", "LQR", args.max_bellman_iterations, args.seed, train=False)
        p = json.load(open(f"experiments/lqr/figures/{args.experiment_name}/parameters.json"))  # p for parameters

        from experiments.lqr.utils import define_environment, define_q
        from pbo.networks.learnable_pbo import CustomLinearPBO

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["max_discrete_state"])

        q = define_q(p["n_actions_on_max"], p["max_action_on_max"], jax.random.PRNGKey(0))
        pbo = CustomLinearPBO(
            q=q,
            max_bellman_iterations=args.max_bellman_iterations,
            network_key=jax.random.PRNGKey(0),
            learning_rate={"first": 0, "last": 0, "duration": 0},
            initial_weight_std=0.1,
        )
        pbo.params["CustomLinearPBONet"]["slope"] = env.optimal_slope.reshape((1, 3))
        pbo.params["CustomLinearPBONet"]["bias"] = env.optimal_bias.reshape((1, 3))

        weights = np.zeros(
            (
                args.max_bellman_iterations + args.validation_bellman_iterations + 1,
                q.weights_dimension,
            )
        )

        q_weights = q.to_weights(q.params)

        for iteration in tqdm(range(args.max_bellman_iterations + args.validation_bellman_iterations + 1)):
            weights[iteration] = q_weights
            q_weights = pbo(pbo.params, q_weights.reshape((1, -1)))[0]

        np.save(
            f"experiments/lqr/figures/{args.experiment_name}/PBO_optimal/{args.max_bellman_iterations}_W.npy",
            weights,
        )
        np.save(
            f"experiments/lqr/figures/{args.experiment_name}/PBO_optimal/{args.max_bellman_iterations}_V.npy",
            env.greedy_V(weights),
        )
