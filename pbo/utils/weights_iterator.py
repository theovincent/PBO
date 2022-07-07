import time
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
        n_iterations: int,
        add_infinity: bool,
        sleeping_time: float,
    ) -> None:
        self.pbo = pbo
        self.weights = weights
        self.n_iterations = n_iterations
        self.add_infinity = add_infinity

        self.optimal_fixed_point = pbo_optimal.fixed_point()
        self.pbo_on_weights_fixed_point = pbo_on_weights.fixed_point(pbo_on_weights.params)
        self.pbo_on_weighs_fixed_point_error = np.sqrt(
            np.square(self.optimal_fixed_point - self.pbo_on_weights_fixed_point).mean()
        )

        self.iterated_weights = np.zeros((n_iterations + 1, weights.shape[0], weights.shape[1]))
        self.iterated_weights[0] = weights

        self.iterated_weights_optimal = np.zeros((n_iterations + 1, weights.shape[0], weights.shape[1]))
        self.iterated_weights_optimal[0] = weights
        self.iterated_weights_pbo_on_weights = np.zeros((n_iterations + 1, weights.shape[0], weights.shape[1]))
        self.iterated_weights_pbo_on_weights[0] = weights

        for iteration in range(1, self.n_iterations + 1):
            self.iterated_weights_optimal[iteration] = pbo_optimal(self.iterated_weights_optimal[iteration - 1])
            self.iterated_weights_pbo_on_weights[iteration] = pbo_on_weights(
                pbo_on_weights.params, self.iterated_weights_pbo_on_weights[iteration - 1]
            )

        self.iterated_pbo_on_weights_error = np.sqrt(
            np.square(self.iterated_weights_optimal - self.iterated_weights_pbo_on_weights).mean(axis=(1, 2))
        )

        self.sleeping_time = sleeping_time

    def iterate_on_params(self, params: hk.Params, fixed_point: jnp.ndarray) -> None:
        for iteration in range(1, self.n_iterations + 1):
            self.iterated_weights[iteration] = self.pbo(params, self.iterated_weights[iteration - 1])

        self.fixed_point = fixed_point

    def show(self, title: str = "") -> None:
        clear_output(wait=True)

        # Plot the iterations on the first weight
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 5))
        labels = ["k", "i", " m"]

        for idx_ax, ax in enumerate(axes):
            ax.axhline(y=self.optimal_fixed_point[idx_ax], color="black", label="Optimal fixed point")
            ax.axhline(
                y=self.pbo_on_weights_fixed_point[idx_ax],
                color="grey",
                label="PBO on weights fixed point",
                linestyle="--",
            )
            ax.axhline(y=self.fixed_point[idx_ax], color="g", label="fixed point", linestyle="--")
            ax.scatter(
                range(self.n_iterations + 1),
                self.iterated_weights_optimal[:, 0, idx_ax],
                color="black",
                label="Optimally iterated weights",
            )
            ax.scatter(
                range(self.n_iterations + 1),
                self.iterated_weights_pbo_on_weights[:, 0, idx_ax],
                color="grey",
                label="PBO on weights iterated weights",
                marker="x",
            )
            ax.scatter(
                range(self.n_iterations + 1),
                self.iterated_weights[:, 0, idx_ax],
                color="g",
                label="Iterated weights",
                marker="x",
            )

            ax.set_ylim(
                self.optimal_fixed_point[idx_ax] - 2 * abs(self.optimal_fixed_point[idx_ax]),
                self.optimal_fixed_point[idx_ax] + 2 * abs(self.optimal_fixed_point[idx_ax]),
            )
            ax.set_ylabel(labels[idx_ax])
            if idx_ax != 2:
                ax.set_xticks([])
                ax.set_xticklabels([])
            else:
                ax.set_xticks(range(self.n_iterations + 1))
                ax.set_xticklabels(range(self.n_iterations + 1))

        if title != "":
            axes[0].set_title(title)
        axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        axes[2].set_xlabel("iterations")

        fig.tight_layout()
        fig.canvas.draw()

        # Plot the errors on all weights
        plt.figure(figsize=(7, 3))

        iterated_error = np.sqrt(np.square(self.iterated_weights_optimal - self.iterated_weights).mean(axis=(1, 2)))

        plt.bar(
            range(self.n_iterations + 1),
            iterated_error,
            color="g",
            label=r"$||T^*(w) - T_{linear}(w)||$",
        )
        plt.bar(
            range(self.n_iterations + 1),
            self.iterated_pbo_on_weights_error,
            color="grey",
            label=r"$||T^*(w) - T_{linear\_on\_weights}(w)||$",
            alpha=0.9,
        )
        plt.axhline(
            y=np.square(self.optimal_fixed_point - self.fixed_point).mean(),
            color="g",
            label="||opt_fixed_point - fixed_point||",
        )
        plt.axhline(
            y=self.pbo_on_weighs_fixed_point_error,
            color="grey",
            label="||opt_fixed_point - fixed_point_linear_on_weights||",
        )

        plt.ylabel("errors")
        plt.xlabel("iteration")
        not_nan = tuple(~jnp.isnan(iterated_error))
        plt.ylim(0, iterated_error[not_nan].max() + 0.1)

        pbo_total_error = iterated_error.sum()
        pbo_on_weights_total_error = self.iterated_pbo_on_weights_error.sum()
        if self.add_infinity:
            pbo_total_error += np.sqrt(np.square(self.optimal_fixed_point - self.fixed_point).mean())
            pbo_on_weights_total_error += self.pbo_on_weighs_fixed_point_error
        plt.title(
            f"PBO total error: {str(jnp.round(pbo_total_error, 2))}, PBO on weights total error: {str(jnp.round(pbo_on_weights_total_error, 2))}"
        )
        plt.legend()
        plt.show()

        time.sleep(self.sleeping_time)

    @staticmethod
    def add_points(ax, points: np.ndarray, size: float, label: str, color: str) -> None:
        xdata = points[:, 0]
        ydata = points[:, 1]
        zdata = points[:, 2]
        ax.scatter3D(xdata, ydata, zdata, s=size, label=label, color=color)

    def visualize(self, pbo: BasePBO, optimal: bool) -> None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        sizes = [1, 5, 300, 1000]
        colors = ["black", "b", "red", "g"]
        iterated_weights = self.weights

        for iteration in range(4):
            self.add_points(ax, iterated_weights, sizes[iteration], f"iteration {iteration}", colors[iteration])
            if optimal:
                iterated_weights = pbo(iterated_weights)
            else:
                iterated_weights = pbo(pbo.params, iterated_weights)

        ax.set_xlabel("k")
        ax.set_xticklabels([])
        ax.set_xticks([])

        ax.set_ylabel("i")
        ax.set_yticklabels([])
        ax.set_yticks([])

        ax.set_zlabel("m")
        ax.set_zlim(-2, 5)

        ax.legend()
        ax.view_init(0, 0)
        fig.tight_layout()
        plt.show()
