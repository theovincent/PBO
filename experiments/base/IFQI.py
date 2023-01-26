from tqdm import tqdm
import numpy as np
from argparse import Namespace

from pbo.sample_collection.dataloader import SampleDataLoader
from pbo.networks.base_q import BaseQ
from pbo.utils.params import save_params


def train(environment_name: str, args: Namespace, q: BaseQ, p: dict, data_loader_samples: SampleDataLoader) -> None:
    l2_losses = np.ones((p["training_steps_ifqi"], p["fitting_steps_ifqi"])) * np.nan

    for training_step in tqdm(range(p["training_steps_ifqi"])):
        params_target = q.params

        for fitting_step in tqdm(range(p["fitting_steps_ifqi"]), leave=False):
            cumulative_l2_loss = 0

            data_loader_samples.shuffle()
            for batch_samples in data_loader_samples:
                q.params, q.optimizer_state, l2_loss = q.learn_on_batch(
                    q.params, params_target, q.optimizer_state, batch_samples
                )
                cumulative_l2_loss += l2_loss

            l2_losses[training_step, fitting_step] = cumulative_l2_loss

    save_params(
        f"experiments/{environment_name}/figures/{args.experiment_name}/IFQI/{args.max_bellman_iterations}_P_{args.seed}",
        q.params,
    )
    np.save(
        f"experiments/{environment_name}/figures/{args.experiment_name}/IFQI/{args.max_bellman_iterations}_L_{args.seed}.npy",
        l2_losses,
    )
