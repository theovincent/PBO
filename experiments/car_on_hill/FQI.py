import sys
import argparse
import json
import numpy as np
import jax
from tqdm import tqdm


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train FQI on Car-On-Hill.")
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
    print(f"Training FQI on Car-On-Hill with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}...")
    p = json.load(open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.car_on_hill.utils import define_environment
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.sample_collection.dataloader import SampleDataLoader
    from pbo.networks.learnable_q import FullyConnectedQ
    from pbo.utils.params import save_params

    key = jax.random.PRNGKey(args.seed)
    shuffle_key, q_network_key, _ = jax.random.split(
        key, 3
    )  # 3 keys are generated to be coherent with the other trainings

    env, _, _, _, _ = define_environment(p["gamma"], p["n_states_x"], p["n_states_v"])

    replay_buffer = ReplayBuffer()
    replay_buffer.load("experiments/car_on_hill/figures/data/replay_buffer.npz")
    data_loader_samples = SampleDataLoader(replay_buffer, p["batch_size_samples"], shuffle_key)

    q = FullyConnectedQ(
        state_dim=2,
        action_dim=1,
        actions_on_max=env.actions_on_max,
        gamma=p["gamma"],
        network_key=q_network_key,
        layers_dimension=p["layers_dimension"],
        zero_initializer=True,
        learning_rate={
            "first": p["starting_lr_fqi"],
            "last": p["ending_lr_fqi"],
            "duration": p["fitting_steps_fqi"] * len(replay_buffer) // p["batch_size_samples"],
        },
    )

    l2_losses = np.ones((args.max_bellman_iterations, p["fitting_steps_fqi"])) * np.nan
    iterated_params = {}
    iterated_params["0"] = q.params

    for bellman_iteration in tqdm(range(1, args.max_bellman_iterations + 1)):
        q.reset_optimizer()
        params_target = q.params
        best_loss = float("inf")
        patience = 0

        for step in range(p["fitting_steps_fqi"]):
            cumulative_l2_loss = 0

            data_loader_samples.shuffle()
            for batch_samples in data_loader_samples:
                q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
                    q.params, params_target, q.optimizer_state, batch_samples
                )
                cumulative_l2_loss += l2_loss

            l2_losses[bellman_iteration - 1, step] = cumulative_l2_loss
            if cumulative_l2_loss < best_loss:
                patience = 0
                best_loss = cumulative_l2_loss
            else:
                patience += 1

            if patience > p["patience"]:
                break

        iterated_params[f"{bellman_iteration}"] = q.params

    save_params(
        f"experiments/car_on_hill/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_P_{args.seed}",
        iterated_params,
    )
    np.save(
        f"experiments/car_on_hill/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )
