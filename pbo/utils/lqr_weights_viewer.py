import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp

from pbo.networks.base_pbo import BasePBO


def add_points(ax, points: np.ndarray, label: str) -> None:
    xdata = points[..., 0]
    ydata = points[..., 1]
    zdata = points[..., 2]
    ax.scatter3D(xdata, ydata, zdata, label=label)


def visualize(
    weights: jnp.ndarray, optimal_weights: jnp.ndarray, fixed_point: jnp.ndarray = None, title: str = ""
) -> None:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    for iteration in range(weights.shape[0]):
        add_points(ax, weights[iteration], f"iteration {iteration}")

    if fixed_point is not None:
        ax.scatter3D(
            fixed_point[0],
            fixed_point[1],
            fixed_point[2],
            color="black",
            label=r"iteration $+\infty$",
            s=80,
        )
    ax.scatter3D(
        optimal_weights[0],
        optimal_weights[1],
        optimal_weights[2],
        color="black",
        label=r"optimal iteration $+\infty$",
        s=100,
        marker="*",
    )

    ax.set_xlabel("K")
    ax.set_ylabel("I")
    ax.set_zlabel("M")
    if len(weights.shape) > 2:
        ax.set_xlim(min(weights[0, :, 0]), max(weights[0, :, 0]))
        ax.set_ylim(min(weights[0, :, 1]), max(weights[0, :, 1]))
        ax.set_zlim(min(weights[0, :, 2]), -min(weights[0, :, 2]))

    ax.legend()
    ax.view_init(5, 50)
    plt.title(title)
    fig.tight_layout()
    plt.show()
