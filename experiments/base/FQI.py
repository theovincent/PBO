from tqdm import tqdm
import numpy as np
from argparse import Namespace

from pbo.sample_collection.dataloader import SampleDataLoader
from pbo.networks.base_q import BaseQ
from pbo.utils.params import save_params


def train(environment_name: str, args: Namespace, q: BaseQ, p: dict, data_loader_samples: SampleDataLoader) -> None:
    l2_losses = np.ones((args.max_bellman_iterations, p["fitting_steps_fqi"])) * np.nan
    iterated_params = {}
    iterated_params["0"] = q.params

    for bellman_iteration in tqdm(range(1, args.max_bellman_iterations + 1)):
        q.reset_optimizer()
        params_target = q.params
        best_loss = float("inf")
        patience = 0

        for step in tqdm(range(p["fitting_steps_fqi"]), leave=False):
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
        f"experiments/{environment_name}/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_P_{args.seed}",
        iterated_params,
    )
    np.save(
        f"experiments/{environment_name}/figures/{args.experiment_name}/FQI/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )
