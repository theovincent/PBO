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

        parser = argparse.ArgumentParser("Compute PBO optimal on Chain Walk.")
        addparse(parser, validation_bellman_iterations=True)
        args = parser.parse_args(argvs)
        print_info(args.experiment_name, "an optimal PBO", "Chain Walk", args.max_bellman_iterations, train=False)
        p = json.load(
            open(f"experiments/chain_walk/figures/{args.experiment_name}/parameters.json")
        )  # p for parameters

        from experiments.chain_walk.utils import define_environment, define_q
        from pbo.networks.learnable_pbo import MaxLinearPBO

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["n_states"], p["sucess_probability"], p["gamma"])

        q = define_q(p["n_states"], env.n_actions, p["gamma"], jax.random.PRNGKey(0))
        pbo = MaxLinearPBO(
            q=q,
            max_bellman_iterations=args.max_bellman_iterations,
            network_key=jax.random.PRNGKey(0),
            learning_rate={"first": 0, "last": 0, "duration": 0},
            n_actions=env.n_actions,
            initial_weight_std=0,
        )
        pbo.params["MaxLinearPBONet/linear"]["w"] = p["gamma"] * env.transition_proba.T
        pbo.params["MaxLinearPBONet/linear"]["b"] = env.R.T

        iterated_q_estimate = np.zeros(
            (
                args.max_bellman_iterations + args.validation_bellman_iterations + 1,
                env.n_states,
                env.n_actions,
            )
        )
        iterated_v = np.zeros((args.max_bellman_iterations + args.validation_bellman_iterations + 1, env.n_states))
        batch_iterated_weights = q.to_weights(q.params).reshape((1, -1))

        for iteration in tqdm(range(args.max_bellman_iterations + args.validation_bellman_iterations + 1)):
            iterated_q_estimate[iteration] = env.discretize(q, q.to_params(batch_iterated_weights[0]))
            policy_q = iterated_q_estimate[iteration].argmax(axis=1)
            iterated_v[iteration] = env.value_function(policy_q)

            batch_iterated_weights = pbo(pbo.params, batch_iterated_weights)

        np.save(
            f"experiments/chain_walk/figures/{args.experiment_name}/PBO_optimal/{args.max_bellman_iterations}_Q.npy",
            iterated_q_estimate,
        )
        np.save(
            f"experiments/chain_walk/figures/{args.experiment_name}/PBO_optimal/{args.max_bellman_iterations}_V.npy",
            iterated_v,
        )
