import sys
import argparse
import json
import jax
import jax.numpy as jnp
from tqdm import tqdm


def run_cli(argvs=sys.argv[1:]):
    with jax.default_device(jax.devices("cpu")[0]):
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        parser = argparse.ArgumentParser("Collect sample on Bicycle.")
        parser.add_argument(
            "-e",
            "--experiment_name",
            help="Experiment name.",
            type=str,
            required=True,
        )
        args = parser.parse_args(argvs)
        print(f"{args.experiment_name}:")
        p = json.load(
            open(f"experiments/bicycle_offline/figures/{args.experiment_name}/parameters.json")
        )  # p for parameters
        print(f"Collecting {p['n_samples']} samples on Bicycle...")

        from experiments.bicycle_offline.utils import define_environment
        from pbo.sample_collection.replay_buffer import ReplayBuffer

        sample_key = jax.random.PRNGKey(p["env_seed"])

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma"])

        replay_buffer = ReplayBuffer(p["n_samples"])

        env.reset()
        n_episodes = 0
        n_steps = 0

        for _ in tqdm(range(p["n_samples"])):
            state = env.state

            sample_key, key = jax.random.split(sample_key)
            action = jax.random.choice(key, env.actions_on_max)

            next_state, reward, absorbing, _ = env.step(action)
            n_steps += 1

            replay_buffer.add(state, action, reward, next_state, absorbing)

            if absorbing[0] or n_steps >= 20:
                sample_key, key = jax.random.split(sample_key)
                env.reset(
                    jax.random.multivariate_normal(
                        key,
                        jnp.zeros(4),
                        jnp.array([[1e-4, -1e-4, 0, 0], [-1e-4, 1e-3, 0, 0], [0, 0, 1e-3, -1e-4], [0, 0, -1e-4, 1e-2]]),
                    )
                    / 10
                )
                n_episodes += 1
                n_steps = 0

        print(f"Number of episodes: {n_episodes}")

        replay_buffer.save(f"experiments/bicycle_offline/figures/{args.experiment_name}/replay_buffer.npz")
