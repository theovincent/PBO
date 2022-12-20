import sys
import argparse
import json
import numpy as np
import jax
from tqdm import tqdm


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
        help="Maximum number of Bellman iteration.",
        type=int,
        required=True,
    )
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    print(f"Training DQN on Lunar Lander with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}...")
    p = json.load(open(f"experiments/lunar_lander/figures/{args.experiment_name}/parameters.json"))  # p for parameters
    print(
        f"The target parameters are changed every {p['training_updates_dqn'] // args.max_bellman_iterations} updates."
    )

    from experiments.lunar_lander.utils import define_environment, collect_random_samples, collect_sample
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.sample_collection.dataloader import SampleDataLoader
    from pbo.networks.learnable_q import FullyConnectedQ
    from pbo.utils.params import save_params

    key = jax.random.PRNGKey(args.seed)
    shuffle_key, q_network_key, _ = jax.random.split(
        key, 3
    )  # 3 keys are generated to be coherent with the other trainings

    env = define_environment(jax.random.PRNGKey(p["env_key"]), p["gamma"])

    replay_buffer = ReplayBuffer()
    collect_random_samples(env, replay_buffer, p["n_initial_samples"], p["horizon"])
    data_loader_samples = SampleDataLoader(replay_buffer, p["batch_size_samples"], shuffle_key)

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
            "duration": p["training_updates_dqn"],
        },
    )

    l2_losses = np.ones(p["training_updates_dqn"]) * np.nan
    iterated_params = {}
    iterated_params["0"] = q.params
    params_target = q.params

    for update in range(p["training_updates_dqn"]):
        collect_sample(env, replay_buffer, q, q.params, p["steps_updates_dqn"])
        data_loader_samples = SampleDataLoader(replay_buffer, p["batch_size_samples"], shuffle_key)

        data_loader_samples.shuffle()
        q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
            q.params, params_target, q.optimizer_state, data_loader_samples[0]
        )

        l2_losses[update] = l2_loss

        if (update + 1) % (p["training_updates_dqn"] // args.max_bellman_iterations) == 0:
            iterated_params[f"{(epoch * p['fitting_steps_dqn'] + step + 1) // p['update_target_dqn']}"] = q.params
            params_target = q.params

    save_params(
        f"experiments/lunar_lander/figures/{args.experiment_name}/DQN/{args.max_bellman_iterations}_P_{args.seed}",
        iterated_params,
    )
    np.save(
        f"experiments/lunar_lander/figures/{args.experiment_name}/DQN/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )
