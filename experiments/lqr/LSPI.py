import sys
import argparse
import json
import numpy as np
import jax
from tqdm import tqdm


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train LSPI on LQR.")
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
    print(f"Training LSPI on LQR with {args.max_bellman_iterations} Bellman iterationsand seed {args.seed}...")
    p = json.load(open(f"experiments/lqr/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.lqr.utils import define_environment
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.sample_collection.dataloader import SampleDataLoader
    from pbo.networks.learnable_q import LQRQ
    from pbo.utils.params import save_params

    key = jax.random.PRNGKey(args.seed)
    initial_policy_key, _, _ = jax.random.split(key, 3)  # 3 keys are generated to be coherent with the other trainings

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["max_discrete_state"])

    replay_buffer = ReplayBuffer(p["n_discrete_states"] * p["n_discrete_actions"])
    replay_buffer.load(f"experiments/lqr/figures/{args.experiment_name}/replay_buffer.npz")
    data_loader_samples = SampleDataLoader(replay_buffer, 1, None)

    q = LQRQ(
        n_actions_on_max=p["n_actions_on_max"],
        max_action_on_max=p["max_action_on_max"],
        network_key=jax.random.PRNGKey(0),
        zero_initializer=True,
    )

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
