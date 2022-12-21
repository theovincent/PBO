import sys
import argparse
from tqdm import tqdm
import json
import numpy as np
import jax
import optax


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train DQN on Lunar Lander.")
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
        help="Maximum number of Bellman iterations.",
        type=int,
        required=True,
    )
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    print(f"Training DQN on Lunar Lander with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}...")
    p = json.load(open(f"experiments/lunar_lander/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.lunar_lander.utils import define_environment, collect_random_samples, collect_samples
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.sample_collection.dataloader import SampleDataLoader
    from pbo.networks.learnable_q import FullyConnectedQ
    from pbo.utils.params import save_params

    key = jax.random.PRNGKey(args.seed)
    sample_key, q_network_key, _ = jax.random.split(
        key, 3
    )  # 3 keys are generated to be coherent with the other trainings

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

    replay_buffer = ReplayBuffer(p["max_size"])
    collect_random_samples(env, replay_buffer, p["n_initial_samples"], p["horizon"])

    q = FullyConnectedQ(
        state_dim=8,
        action_dim=1,
        actions_on_max=env.actions_on_max,
        gamma=p["gamma"],
        network_key=q_network_key,
        layers_dimension=p["layers_dimension"],
        zero_initializer=True,
        learning_rate={
            "first": p["starting_lr_dqn"],
            "last": p["ending_lr_dqn"],
            "duration": args.max_bellman_iterations * p["fitting_updates_dqn"],
        },
    )
    epsilon_schedule = optax.linear_schedule(
        p["starting_eps_dqn"], p["ending_eps_dqn"], args.max_bellman_iterations * p["fitting_updates_dqn"]
    )

    l2_losses = np.ones((args.max_bellman_iterations, p["fitting_updates_dqn"])) * np.nan
    iterated_params = {}
    iterated_params["0"] = q.params

    for bellman_iteration in tqdm(range(1, args.max_bellman_iterations + 1)):
        params_target = q.params

        for update in tqdm(range(p["fitting_updates_dqn"]), leave=False):
            collect_samples(
                env,
                replay_buffer,
                q,
                q.params,
                p["steps_per_update_dqn"],
                p["horizon"],
                epsilon_schedule((bellman_iteration - 1) * p["fitting_updates_dqn"] + update),
            )

            sample_key, key = jax.random.split(sample_key)
            batch_data = replay_buffer.sample_random_batch(sample_key, p["batch_size_samples"])

            q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
                q.params, params_target, q.optimizer_state, batch_data
            )

            l2_losses[bellman_iteration - 1, update] = l2_loss

        iterated_params[f"{bellman_iteration}"] = q.params

    save_params(
        f"experiments/lunar_lander/figures/{args.experiment_name}/DQN/{args.max_bellman_iterations}_P_{args.seed}",
        iterated_params,
    )
    np.save(
        f"experiments/lunar_lander/figures/{args.experiment_name}/DQN/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )
