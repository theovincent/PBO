import json
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm


def run_cli():
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    p = json.load(open("experiments/car_on_hill/parameters.json"))  # p for parameters
    print(f"Collecting {p['n_random_samples'] + p['n_oriented_samples']} samples on Car-On-Hill...")

    from experiments.car_on_hill.utils import define_environment
    from pbo.sample_collection.replay_buffer import ReplayBuffer
    from pbo.sample_collection.count_samples import count_samples

    env, _, states_x_boxes, _, states_v_boxes = define_environment(p["gamma"], p["n_states_x"], p["n_states_v"])
    sample_key = jax.random.PRNGKey(p["env_seed"])

    replay_buffer = ReplayBuffer()

    env.reset()
    n_episodes = 0
    n_steps = 0
    for _ in tqdm(range(p["n_random_samples"])):
        state = env.state

        sample_key, key = jax.random.split(sample_key)
        action = jax.random.choice(key, env.actions_on_max)
        next_state, reward, absorbing, _ = env.step(action)
        n_steps += 1

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or n_steps > p["horizon"]:
            env.reset()
            n_episodes += 1
            n_steps = 0

    env.reset(jnp.array(p["oriented_states"])[1])
    n_episodes += 1
    n_steps = 0
    for _ in tqdm(range(p["n_oriented_samples"])):
        state = env.state

        sample_key, key = jax.random.split(sample_key)
        action = jax.random.choice(key, env.actions_on_max)
        next_state, reward, absorbing, _ = env.step(action)
        n_steps += 1

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or n_steps > p["horizon"]:
            sample_key, key = jax.random.split(sample_key)
            alpha = jax.random.uniform(key)
            env.reset(alpha * jnp.array(p["oriented_states"])[0] + (1 - alpha) * jnp.array(p["oriented_states"])[1])

            n_episodes += 1
            n_steps = 0

    assert sum(jnp.array(replay_buffer.rewards) == 1) > 0, "No positive reward has been sampled, please do something!"
    print(f"Number of episodes: {n_episodes}")

    replay_buffer.save("experiments/car_on_hill/figures/data/replay_buffer.npz")

    replay_buffer.cast_to_jax_array()
    samples_count, _, _ = count_samples(
        replay_buffer.states[:, 0], replay_buffer.states[:, 1], states_x_boxes, states_v_boxes, replay_buffer.rewards
    )
    np.save(f"experiments/car_on_hill/figures/data/samples_count.npy", samples_count)
