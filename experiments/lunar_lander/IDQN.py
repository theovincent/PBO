import sys
import argparse
import json
import numpy as np
import jax
from tqdm import tqdm
import optax


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train IDQN on Lunar Lander.")
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
    print(
        f"Training IDQN on Lunar Lander with {args.max_bellman_iterations} Bellman iterations and seed {args.seed}..."
    )
    p = json.load(open(f"experiments/lunar_lander/figures/{args.experiment_name}/parameters.json"))  # p for parameters

    from experiments.lunar_lander.utils import define_environment, collect_random_samples, collect_samples
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.networks.learnable_multi_head_q import FullyConnectedMultiHeadQ
    from pbo.utils.params import save_params

    key = jax.random.PRNGKey(args.seed)
    sample_key, q_network_key, _ = jax.random.split(
        key, 3
    )  # 3 keys are generated to be coherent with the other trainings

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

    replay_buffer = ReplayBuffer(p["max_size"])
    collect_random_samples(env, replay_buffer, p["n_initial_samples"], p["horizon"])

    q = FullyConnectedMultiHeadQ(
        n_heads=args.max_bellman_iterations + 1,
        state_dim=8,
        action_dim=1,
        actions_on_max=env.actions_on_max,
        gamma=p["gamma"],
        network_key=q_network_key,
        layers_dimension=p["layers_dimension"],
        zero_initializer=True,
        learning_rate={
            "first": p["starting_lr_idqn"],
            "last": p["ending_lr_idqn"],
            "duration": p["training_steps_idqn"] * p["fitting_updates_idqn"],
        },
    )
    epsilon_schedule = optax.linear_schedule(
        p["starting_eps_idqn"], p["ending_eps_idqn"], args.max_bellman_iterations * p["fitting_updates_idqn"]
    )

    l2_losses = np.ones((p["training_steps_idqn"], p["fitting_updates_idqn"])) * np.nan

    for training_step in tqdm(range(p["training_steps_idqn"])):
        params_target = q.params

        for fitting_step in range(p["fitting_updates_idqn"]):
            q_inference = jax.jit(
                lambda q_params_, state_, action_: q(q_params_, state_, action_)[..., args.max_bellman_iterations]
            )

            collect_samples(
                env,
                replay_buffer,
                q_inference,
                q.params,
                p["steps_per_update"],
                p["horizon"],
                epsilon_schedule(training_step * p["fitting_updates_idqn"] + fitting_step),
            )

            sample_key, key = jax.random.split(sample_key)
            batch_samples = replay_buffer.sample_random_batch(sample_key, p["batch_size_samples"])

            q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
                q.params, params_target, q.optimizer_state, batch_samples
            )

            l2_losses[training_step, fitting_step] = l2_loss

    save_params(
        f"experiments/lunar_lander/figures/{args.experiment_name}/IDQN/{args.max_bellman_iterations}_P_{args.seed}",
        q.params,
    )
    np.save(
        f"experiments/lunar_lander/figures/{args.experiment_name}/IDQN/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )
