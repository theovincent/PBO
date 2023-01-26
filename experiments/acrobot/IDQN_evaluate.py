import sys
import argparse
import multiprocessing
import json
import jax
import jax.numpy as jnp
import numpy as np

from experiments.base.parser import addparse
from experiments.base.print import print_info


def run_cli(argvs=sys.argv[1:]):
    with jax.default_device(jax.devices("cpu")[0]):
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)

        parser = argparse.ArgumentParser("Evaluate a IDQN on Acrobot.")
        addparse(parser, seed=True)
        args = parser.parse_args(argvs)
        print_info(args.experiment_name, "IDQN", "Acrobot", args.max_bellman_iterations, args.seed, train=False)
        p = json.load(open(f"experiments/acrobot/figures/{args.experiment_name}/parameters.json"))  # p for parameters

        from experiments.acrobot.utils import define_environment, define_q_multi_head
        from pbo.networks.learnable_multi_head_q import FullyConnectedMultiHeadQ
        from pbo.utils.params import load_params

        env = define_environment(jax.random.PRNGKey(p["env_seed"]), p["gamma_evaluation"])

        q = define_q_multi_head(
            args.max_bellman_iterations + 1,
            env.actions_on_max,
            p["gamma"],
            jax.random.PRNGKey(0),
            p["layers_dimension"],
        )
        q.params = load_params(
            f"experiments/acrobot/figures/{args.experiment_name}/IDQN/{args.max_bellman_iterations}_P_{args.seed}"
        )

        def evaluate(
            iteration: int,
            j_list: list,
            q: FullyConnectedMultiHeadQ,
            q_weights: jnp.ndarray,
            horizon: int,
        ):
            q_inference = jax.jit(lambda q_params_, state_, action_: q(q_params_, state_, action_)[..., iteration])

            j_list[iteration] = env.evaluate(
                q_inference,
                q.to_params(q_weights),
                horizon,
                p["n_simulations"],
                video_path=f"{args.experiment_name}/IDQN/{iteration}_{args.seed}",
            )

        manager = multiprocessing.Manager()
        iterated_j = manager.list(list(np.nan * np.zeros(args.max_bellman_iterations + 1)))

        processes = []
        for iteration in range(args.max_bellman_iterations + 1):
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(iteration, iterated_j, q, q.to_weights(q.params), p["horizon_evaluation"]),
                )
            )

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        np.save(
            f"experiments/acrobot/figures/{args.experiment_name}/IDQN/{args.max_bellman_iterations}_J_{args.seed}.npy",
            iterated_j,
        )
