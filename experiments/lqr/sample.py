import sys
import argparse
import json
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


def run_cli(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Collect sample on LQR.")
    parser.add_argument(
        "-e",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    args = parser.parse_args(argvs)
    print(f"{args.experiment_name}:")
    p = json.load(open(f"experiments/lqr/figures/{args.experiment_name}/parameters.json"))  # p for parameters
    print("Collecting samples on LQR...")

    from experiments.lqr.utils import define_environment
    from pbo.sample_collection.replay_buffer import ReplayBuffer

    env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["max_discrete_state"])

    replay_buffer = ReplayBuffer(p["n_discrete_states"] * p["n_discrete_actions"])

    for state in tqdm(np.linspace(-p["max_discrete_state"], p["max_discrete_state"], p["n_discrete_states"])):
        for action in np.linspace(-p["max_discrete_action"], p["max_discrete_action"], p["n_discrete_actions"]):
            env.reset(jnp.array([state]))
            next_state, reward, absorbing, _ = env.step(jnp.array([action]))

            replay_buffer.add(jnp.array([state]), jnp.array([action]), reward, next_state, absorbing)

    replay_buffer.save(f"experiments/lqr/figures/{args.experiment_name}/replay_buffer.npz")
