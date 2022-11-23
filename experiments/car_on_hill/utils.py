import os
import shutil
from typing import Tuple
import numpy as np

from pbo.environments.car_on_hill import CarOnHillEnv


def define_environment(
    gamma: float, n_states_x: int, n_states_v: int
) -> Tuple[CarOnHillEnv, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    env = CarOnHillEnv(gamma)

    states_x = np.linspace(-env.max_position, env.max_position, n_states_x)
    boxes_x_size = (2 * env.max_position) / (n_states_x - 1)
    states_x_boxes = np.linspace(-env.max_position, env.max_position + boxes_x_size, n_states_x + 1) - boxes_x_size / 2
    states_v = np.linspace(-env.max_velocity, env.max_velocity, n_states_v)
    boxes_v_size = (2 * env.max_velocity) / (n_states_v - 1)
    states_v_boxes = np.linspace(-env.max_velocity, env.max_velocity + boxes_v_size, n_states_v + 1) - boxes_v_size / 2

    return env, states_x, states_x_boxes, states_v, states_v_boxes


def create_experiment_folders(experiment_name):
    if not os.path.exists(f"experiments/car_on_hill/figures/{experiment_name}"):
        os.makedirs(f"experiments/car_on_hill/figures/{experiment_name}/")
        shutil.copyfile(
            "experiments/car_on_hill/parameters.json",
            f"experiments/car_on_hill/figures/{experiment_name}/parameters.json",
        )

        os.mkdir(f"experiments/car_on_hill/figures/{experiment_name}/FQI/")
        os.mkdir(f"experiments/car_on_hill/figures/{experiment_name}/PBO_linear/")
        os.mkdir(f"experiments/car_on_hill/figures/{experiment_name}/PBO_deep/")
        os.mkdir(f"experiments/car_on_hill/figures/{experiment_name}/IFQI/")
