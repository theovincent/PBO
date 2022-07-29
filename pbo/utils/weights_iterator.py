import time
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import jax.numpy as jnp
import haiku as hk

from pbo.networks.base_pbo import BasePBO


class WeightsIterator:
    def __init__(
        self,
        pbo: BasePBO,
        pbo_optimal: BasePBO,
        pbo_on_weights: BasePBO,
        weights: jnp.ndarray,
        max_bellman_iterations: int,
        max_bellman_iterations_validation: int,
        add_infinity: bool,
        sleeping_time: float,
    ) -> None:
        self.pbo = pbo
        self.weights = weights
        self.max_bellman_iterations = max_bellman_iterations
        self.max_bellman_iterations_validation = max_bellman_iterations_validation
        self.add_infinity = add_infinity

        self.optimal_fixed_point = pbo_optimal.fixed_point()
        self.pbo_on_weights_fixed_point = pbo_on_weights.fixed_point(pbo_on_weights.params)
        self.pbo_on_weighs_fixed_point_error = np.sqrt(
            np.square(self.optimal_fixed_point - self.pbo_on_weights_fixed_point).mean()
        )

        self.iterated_weights = np.zeros((max_bellman_iterations_validation + 1, weights.shape[0], weights.shape[1]))
        self.iterated_weights[0] = weights

        self.iterated_weights_optimal = np.zeros(
            (max_bellman_iterations_validation + 1, weights.shape[0], weights.shape[1])
        )
        self.iterated_weights_optimal[0] = weights

        self.iterated_weights_pbo_on_weights = np.zeros(
            (max_bellman_iterations_validation + 1, weights.shape[0], weights.shape[1])
        )
        self.iterated_weights_pbo_on_weights[0] = weights

        for iteration in range(1, self.max_bellman_iterations_validation + 1):
            self.iterated_weights_optimal[iteration] = pbo_optimal(self.iterated_weights_optimal[iteration - 1])
            self.iterated_weights_pbo_on_weights[iteration] = pbo_on_weights(
                pbo_on_weights.params, self.iterated_weights_pbo_on_weights[iteration - 1]
            )

        self.iterated_optimal_error = np.linalg.norm(
            self.optimal_fixed_point - self.iterated_weights_optimal[1:],
            axis=2,
        ).mean(axis=1)
        self.iterated_pbo_on_weights_error = np.linalg.norm(
            self.optimal_fixed_point - self.iterated_weights_pbo_on_weights[1:],
            axis=2,
        ).mean(axis=1)

        self.sleeping_time = sleeping_time

    def iterate_on_params(self, params: hk.Params, fixed_point: jnp.ndarray) -> None:
        for iteration in range(1, self.max_bellman_iterations_validation + 1):
            self.iterated_weights[iteration] = self.pbo(params, self.iterated_weights[iteration - 1])

        self.fixed_point = fixed_point

    def show(self, title: str = "") -> None:
        clear_output(wait=True)

        # Plot the iterations on the first weight
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 5))
        labels = ["K", "I", "M"]

        for idx_ax, ax in enumerate(axes):
            ax.scatter(
                range(self.max_bellman_iterations_validation + 1),
                self.iterated_weights[:, 0, idx_ax],
                color="green",
                label="PBO linear",
                marker="x",
            )
            ax.scatter(
                range(self.max_bellman_iterations_validation + 1),
                self.iterated_weights_pbo_on_weights[:, 0, idx_ax],
                color="grey",
                label="PBO linear on weights",
                marker="x",
            )
            ax.scatter(
                range(self.max_bellman_iterations_validation + 1),
                self.iterated_weights_optimal[:, 0, idx_ax],
                color="black",
                label="PBO optimal",
            )

            ax.hlines(
                y=self.fixed_point[idx_ax],
                xmin=0,
                xmax=self.max_bellman_iterations_validation,
                color="g",
                label="PBO linear fixed point",
                linestyle="--",
            )
            ax.hlines(
                y=self.pbo_on_weights_fixed_point[idx_ax],
                xmin=0,
                xmax=self.max_bellman_iterations_validation,
                color="grey",
                label="PBO linear on weights fixed point",
                linestyle="--",
            )
            ax.hlines(
                y=self.optimal_fixed_point[idx_ax],
                xmin=0,
                xmax=self.max_bellman_iterations_validation,
                color="black",
                label="PBO optimal fixed point",
                linestyle="--",
            )
            ax.axvline(
                x=self.max_bellman_iterations,
                color="black",
                linestyle="--",
            )

            ax.set_ylabel(labels[idx_ax])
            if idx_ax != 2:
                ax.set_xticks([])
                ax.set_xticklabels([])
            else:
                ax.set_xticks(range(self.max_bellman_iterations_validation + 1))
                ax.set_xticklabels(range(self.max_bellman_iterations_validation + 1))

        if title != "":
            axes[0].set_title(title)
        axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        axes[2].set_xlabel("iterations")

        fig.tight_layout()
        fig.canvas.draw()

        # Plot the errors on all weights
        plt.figure(figsize=(7, 3))

        self.iterated_error = np.linalg.norm(
            self.optimal_fixed_point - self.iterated_weights[1:],
            axis=2,
        ).mean(axis=1)
        self.fixed_point_error = np.square(self.optimal_fixed_point - self.fixed_point).mean()

        plt.plot(
            range(1, self.max_bellman_iterations_validation + 1),
            self.iterated_error,
            color="green",
            label="PBO linear",
        )
        plt.plot(
            range(1, self.max_bellman_iterations_validation + 1),
            self.iterated_pbo_on_weights_error,
            color="grey",
            label="PBO linear on weights",
        )
        plt.plot(
            range(1, self.max_bellman_iterations_validation + 1),
            self.iterated_optimal_error,
            color="black",
            label="PBO optimal",
        )
        plt.hlines(
            y=self.fixed_point_error,
            xmin=1,
            xmax=self.max_bellman_iterations_validation,
            color="green",
            linestyle="--",
            label="PBO linear fixed point",
        )
        plt.hlines(
            y=self.pbo_on_weighs_fixed_point_error,
            xmin=1,
            xmax=self.max_bellman_iterations_validation,
            color="grey",
            linestyle="--",
            label="PBO linear on weights fixed point",
        )
        plt.axvline(
            x=self.max_bellman_iterations,
            color="black",
            linestyle="--",
        )

        plt.xlabel("iteration")
        plt.xticks(
            range(1, self.max_bellman_iterations_validation + 1),
            labels=range(1, self.max_bellman_iterations_validation + 1),
        )

        plt.title(r"$E[||w_i - w^*||_2]$")
        plt.legend()
        plt.show()

        time.sleep(self.sleeping_time)

    @staticmethod
    def add_points(ax, points: np.ndarray, color: float, alpha: float, label: str) -> None:
        xdata = points[:, 0]
        ydata = points[:, 1]
        zdata = points[:, 2]
        ax.scatter3D(xdata, ydata, zdata, color=color, alpha=alpha, label=label, s=5)

    def visualize(self, pbo: BasePBO, optimal: bool, title: str = "") -> None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        colors = ["red", "purple", "blue", "green"]
        alphas = [0.1, 0.3, 0.5, 1]
        iterated_weights = self.weights

        for iteration in range(4):
            self.add_points(ax, iterated_weights, colors[iteration], alphas[iteration], f"iteration {iteration}")
            if optimal:
                iterated_weights = pbo(iterated_weights)
            else:
                iterated_weights = pbo(pbo.params, iterated_weights)

        if optimal:
            ax.scatter3D(
                self.optimal_fixed_point[0],
                self.optimal_fixed_point[1],
                self.optimal_fixed_point[2],
                color="black",
                label=r"iteration $+\infty$",
                s=100,
                marker="*",
            )
        else:
            ax.scatter3D(
                pbo.fixed_point(pbo.params)[0],
                pbo.fixed_point(pbo.params)[1],
                pbo.fixed_point(pbo.params)[2],
                color="black",
                label=r"iteration $+\infty$",
                s=80,
            )
            ax.scatter3D(
                self.optimal_fixed_point[0],
                self.optimal_fixed_point[1],
                self.optimal_fixed_point[2],
                color="black",
                label=r"optimal iteration $+\infty$",
                s=100,
                marker="*",
            )

        ax.set_xlabel("k")
        ax.set_xlim(min(self.weights[:, 0]), max(self.weights[:, 0]))
        ax.set_ylabel("i")
        ax.set_ylim(min(self.weights[:, 1]), max(self.weights[:, 1]))
        ax.set_zlabel("m")
        ax.set_zlim(min(self.weights[:, 2]), -min(self.weights[:, 2]))

        ax.legend()
        ax.view_init(0, 10)
        plt.title("Iterations " + title)
        fig.tight_layout()
        plt.show()
