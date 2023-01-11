import sys
import argparse
import json
import numpy as np
import jax
from tqdm import tqdm

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train LSPI on LQR.")
    addparse(parser, seed=True)
    args = parser.parse_args(argvs)
    print_info(args.experiment_name, "LSPI", "LQR", args.max_bellman_iterations)
    p = json.load(open(f"experiments/lqr/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.lqr.utils import define_environment, define_q, define_data_loader_samples, generate_keys
    from pbo.utils.params import save_params

    initial_policy_key, _, _ = generate_keys(args.seed)

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["max_discrete_state"])

    data_loader_samples = define_data_loader_samples(
        p["n_discrete_states"] * p["n_discrete_actions"], args.experiment_name, 1, None
    )
    q = define_q(p["n_actions_on_max"], p["max_action_on_max"], jax.random.PRNGKey(0))

    iterated_params = {}
    iterated_params["0"] = q.params
    weights = np.zeros((args.max_bellman_iterations + 1, q.weights_dimension))

    weights[0] = q.to_weights(q.params)

    for bellman_iteration in tqdm(range(1, args.max_bellman_iterations + 1)):
        A = np.zeros((q.weights_dimension, q.weights_dimension))
        b = np.zeros(q.weights_dimension)

        for batch_samples in tqdm(data_loader_samples, leave=False):
            state = batch_samples["state"][0, 0]
            action = batch_samples["action"][0, 0]
            reward = batch_samples["reward"][0, 0]
            next_state = batch_samples["next_state"][0, 0]
            next_action = q.actions_on_max[q(q.params, next_state, q.actions_on_max).argmax()][0]

            if len(np.unique(q(q.params, next_state, q.actions_on_max))) == 1:
                initial_policy_key, key = jax.random.split(initial_policy_key)
                next_action = jax.random.choice(key, q.actions_on_max.flatten())

            phi = np.zeros((q.weights_dimension, 1))
            phi[0, 0] = state**2
            phi[1, 0] = 2 * state * action
            phi[2, 0] = action**2
            next_phi = np.zeros((q.weights_dimension, 1))
            next_phi[0, 0] = next_state**2
            next_phi[1, 0] = 2 * next_state * next_action
            next_phi[2, 0] = next_action**2

            A += phi @ (phi - next_phi).T
            b += reward * phi.reshape(q.weights_dimension)

        q_weigth_i = np.linalg.solve(A, b)
        q.params = q.to_params(q_weigth_i)

        iterated_params[f"{bellman_iteration}"] = q.params
        weights[bellman_iteration] = q_weigth_i

    save_params(
        f"experiments/lqr/figures/{args.experiment_name}/LSPI/{args.max_bellman_iterations}_P_{args.seed}",
        iterated_params,
    )
    np.save(
        f"experiments/lqr/figures/{args.experiment_name}/LSPI/{args.max_bellman_iterations}_W_{args.seed}.npy",
        weights,
    )
    np.save(
        f"experiments/lqr/figures/{args.experiment_name}/LSPI/{args.max_bellman_iterations}_V_{args.seed}.npy",
        env.greedy_V(weights),
    )
