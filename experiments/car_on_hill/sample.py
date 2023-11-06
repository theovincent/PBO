import sys
import argparse
import json
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Collect sample on Car-On-Hill.")
    parser.add_argument(
        "-e",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    p = json.load(open(f"experiments/car_on_hill/figures/{args.experiment_name}/parameters.json"))  # p for parameters
    print(f"Collecting {p['n_random_samples'] + p['n_oriented_samples']} samples on Car-On-Hill...")

    from pbo.environments.car_on_hill import CarOnHillEnv
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.sample_collection.count_samples import count_samples

    random_sample_key, oriented_sample_key = jax.random.split(jax.random.PRNGKey(p["env_seed"]))

    env = CarOnHillEnv(p["gamma"])

    boxes_x_size = (2 * env.max_position) / (p["n_states_x"] - 1)
    states_x_boxes = (
        np.linspace(-env.max_position, env.max_position + boxes_x_size, p["n_states_x"] + 1) - boxes_x_size / 2
    )
    boxes_v_size = (2 * env.max_velocity) / (p["n_states_v"] - 1)
    states_v_boxes = (
        np.linspace(-env.max_velocity, env.max_velocity + boxes_v_size, p["n_states_v"] + 1) - boxes_v_size / 2
    )

    replay_buffer = ReplayBuffer(p["n_random_samples"] + p["n_oriented_samples"], None, (2,), float, lambda x: x)

    env.reset()
    env.collect_random_samples(random_sample_key, replay_buffer, p["n_random_samples"], p["horizon"])

    env.reset(jnp.array(p["oriented_states"])[1])
    for _ in tqdm(range(p["n_oriented_samples"])):
        state = env.state

        oriented_sample_key, key = jax.random.split(oriented_sample_key)
        action = jax.random.choice(key, jnp.arange(env.n_actions))
        next_state, reward, absorbing, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or env.n_steps >= p["horizon"]:
            oriented_sample_key, key = jax.random.split(oriented_sample_key)
            alpha = jax.random.uniform(key)
            env.reset(alpha * jnp.array(p["oriented_states"])[0] + (1 - alpha) * jnp.array(p["oriented_states"])[1])

    assert sum(jnp.array(replay_buffer.rewards) == 1) > 0, "No positive reward has been sampled, please do something!"
    replay_buffer.save(f"experiments/car_on_hill/figures/{args.experiment_name}/replay_buffer")

    samples_count, _, rewards_count = count_samples(
        replay_buffer.states[:, 0],
        replay_buffer.states[:, 1],
        states_x_boxes,
        states_v_boxes,
        replay_buffer.rewards,
    )
    np.save(f"experiments/car_on_hill/figures/{args.experiment_name}/samples_count.npy", samples_count)
    np.save(f"experiments/car_on_hill/figures/{args.experiment_name}/rewards_count.npy", rewards_count)
