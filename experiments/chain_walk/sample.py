import sys
import argparse
import json
import jax
import jax.numpy as jnp
from tqdm import tqdm


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Collect sample on Chain Walk.")
    parser.add_argument(
        "-e",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    p = json.load(open(f"experiments/chain_walk/figures/{args.experiment_name}/parameters.json"))  # p for parameters
    print("Collecting samples on Chain Walk...")

    from experiments.chain_walk.utils import define_environment
    from pbo.sample_collection.replay_buffer import ReplayBuffer

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["n_states"], p["sucess_probability"], p["gamma"])

    replay_buffer = ReplayBuffer(p["n_states"] * env.n_actions * p["n_repetitions"])

    for state in tqdm(env.states):
        for action in env.actions:
            # Need to repeat the samples to capture the randomness
            for _ in range(p["n_repetitions"]):
                env.reset(jnp.array([state]))
                next_state, reward, absorbing, _ = env.step(jnp.array([action]))

                replay_buffer.add(jnp.array([state]), jnp.array([action]), reward, next_state, absorbing)

    assert sum(jnp.array(replay_buffer.rewards) == 1) > 0, "No positive reward has been sampled, please do something!"

    replay_buffer.save(f"experiments/chain_walk/figures/{args.experiment_name}/replay_buffer.npz")
