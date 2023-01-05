import sys
import argparse
import json
import numpy as np
import jax
from tqdm import tqdm


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train IFQI on Bicycle.")
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
    print(f"Training IFQI on Bicycle with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}...")
    p = json.load(open(f"experiments/bicycle/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.bicycle.utils import define_environment
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.sample_collection.dataloader import SampleDataLoader
    from pbo.networks.learnable_multi_head_q import FullyConnectedMultiHeadQ
    from pbo.utils.params import save_params

    key = jax.random.PRNGKey(args.seed)
    shuffle_key, q_network_key, _ = jax.random.split(
        key, 3
    )  # 3 keys are generated to be coherent with the other trainings

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

    replay_buffer = ReplayBuffer(p["n_samples"])
    replay_buffer.load(f"experiments/bicycle/figures/{args.experiment_name}/replay_buffer.npz")
    data_loader_samples = SampleDataLoader(replay_buffer, p["batch_size_samples"], shuffle_key)

    q = FullyConnectedMultiHeadQ(
        n_heads=args.max_bellman_iterations + 1,
        state_dim=4,
        action_dim=2,
        actions_on_max=env.actions_on_max,
        gamma=p["gamma"],
        network_key=q_network_key,
        layers_dimension=p["layers_dimension"],
        zero_initializer=True,
        learning_rate={
            "first": p["starting_lr_ifqi"],
            "last": p["ending_lr_ifqi"],
            "duration": p["training_steps_ifqi"]
            * p["fitting_steps_ifqi"]
            * replay_buffer.len
            // p["batch_size_samples"],
        },
    )
    l2_losses = np.ones((p["training_steps_ifqi"], p["fitting_steps_ifqi"])) * np.nan

    for training_step in tqdm(range(p["training_steps_ifqi"])):
        params_target = q.params

        for fitting_step in range(p["fitting_steps_ifqi"]):
            cumulative_l2_loss = 0

            data_loader_samples.shuffle()
            for batch_samples in data_loader_samples:
                q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
                    q.params, params_target, q.optimizer_state, batch_samples
                )
                cumulative_l2_loss += l2_loss

            l2_losses[training_step, fitting_step] = cumulative_l2_loss

    save_params(
        f"experiments/bicycle/figures/{args.experiment_name}/IFQI/{args.max_bellman_iterations}_P_{args.seed}",
        q.params,
    )
    np.save(
        f"experiments/bicycle/figures/{args.experiment_name}/IFQI/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )