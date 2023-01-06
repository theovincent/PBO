import sys
import argparse
import json
import numpy as np
import jax
from tqdm import tqdm


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train LSPI on Chain Walk.")
    parser.add_argument(
        "-e",
        "--experiment_name",
        help="Experiment name.",
        type=str,
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
    print(f"Training LSPI on Chain Walk with {args.max_bellman_iterations} Bellman iterations...")
    p = json.load(open(f"experiments/chain_walk/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.chain_walk.utils import define_environment
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.sample_collection.dataloader import SampleDataLoader
    from pbo.networks.learnable_q import TableQ
    from pbo.utils.params import save_params

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["n_states"], p["sucess_probability"], p["gamma"])

    replay_buffer = ReplayBuffer(p["n_states"] * env.n_actions * p["n_repetitions"])
    replay_buffer.load(f"experiments/chain_walk/figures/{args.experiment_name}/replay_buffer.npz")
    data_loader_samples = SampleDataLoader(replay_buffer, 1, None)

    q = TableQ(
        n_states=p["n_states"],
        n_actions=env.n_actions,
        gamma=p["gamma"],
        network_key=jax.random.PRNGKey(0),
        zero_initializer=True,
    )

    iterated_params = {}
    iterated_params["0"] = q.params
    q_functions = np.zeros((args.max_bellman_iterations + 1, env.n_states, env.n_actions))
    v_functions = np.zeros((args.max_bellman_iterations + 1, env.n_states))

    q_i = env.discretize(q, q.params)
    policy_q = q_i.argmax(axis=1)
    q_functions[0] = q_i
    v_functions[0] = env.value_function(policy_q)

    for bellman_iteration in tqdm(range(1, args.max_bellman_iterations + 1)):
        A = np.zeros((q.weights_dimension, q.weights_dimension))
        b = np.zeros(q.weights_dimension)

        for batch_samples in tqdm(data_loader_samples, leave=False):
            state = batch_samples["state"][0, 0]
            action = batch_samples["action"][0, 0]
            reward = batch_samples["reward"][0, 0]
            next_state = batch_samples["next_state"][0, 0]
            next_action = policy_q[next_state]

            phi = np.zeros((q.weights_dimension, 1))
            phi[state * env.n_actions + action, 0] = 1
            next_phi = np.zeros((q.weights_dimension, 1))
            next_phi[next_state * env.n_actions + next_action, 0] = 1

            A += phi @ (phi - p["gamma"] * next_phi).T
            b += reward * phi.reshape(q.weights_dimension)

        q_weigth_i = np.linalg.solve(A, b)
        q_i = env.discretize(q, q.to_params(q_weigth_i))
        policy_q = q_i.argmax(axis=1)

        iterated_params[f"{bellman_iteration}"] = q.to_params(q_weigth_i)
        q_functions[bellman_iteration] = q_i
        v_functions[bellman_iteration] = env.value_function(policy_q)

    save_params(
        f"experiments/chain_walk/figures/{args.experiment_name}/LSPI/{args.max_bellman_iterations}_P",
        iterated_params,
    )
    np.save(
        f"experiments/chain_walk/figures/{args.experiment_name}/LSPI/{args.max_bellman_iterations}_Q.npy",
        q_functions,
    )
    np.save(
        f"experiments/chain_walk/figures/{args.experiment_name}/LSPI/{args.max_bellman_iterations}_V.npy",
        v_functions,
    )
