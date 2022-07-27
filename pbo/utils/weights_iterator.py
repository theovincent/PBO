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
            np.square(self.iterated_weights_optimal[1:] - self.iterated_weights_pbo_on_weights[1:]).mean(axis=(1, 2))
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
            ax.axhline(y=self.optimal_fixed_point[idx_ax], color="black", label=r"$\Gamma^{* w}$")
            ax.axhline(
                y=self.pbo_on_weights_fixed_point[idx_ax],
                color="grey",
                label=r"$\Gamma_{linear\_on\_weights}^w$",
            )
            ax.axhline(y=self.fixed_point[idx_ax], color="g", label=r"$\Gamma_{linear}^w$")
            ax.scatter(
                range(self.n_iterations + 1),
                self.iterated_weights_optimal[:, 0, idx_ax],
                color="black",
                label=r"$\Gamma^{* i}(w_0)$",
            )
            ax.scatter(
                range(self.n_iterations + 1),
                self.iterated_weights_pbo_on_weights[:, 0, idx_ax],
                color="grey",
                label=r"$\Gamma_{linear\_on\_weights}^i(w_0)$",
                marker="x",
            )
            ax.scatter(
                range(self.n_iterations + 1),
                self.iterated_weights[:, 0, idx_ax],
                color="g",
                label=r"$\Gamma_{linear}^i(w_0)$",
                marker="x",
            )

            ax.set_ylim(
                min(
                    self.optimal_fixed_point[idx_ax] - 2 * abs(self.optimal_fixed_point[idx_ax]),
                    self.iterated_weights[0, 0, idx_ax],
                ),
                max(
                    self.optimal_fixed_point[idx_ax] + 2 * abs(self.optimal_fixed_point[idx_ax]),
                    self.iterated_weights[0, 0, idx_ax],
                ),
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

        iterated_error = np.sqrt(
            np.square(self.iterated_weights_optimal[1:] - self.iterated_weights[1:]).mean(axis=(1, 2))
        )

        plt.plot(
            range(1, self.n_iterations + 1),
            iterated_error,
            color="g",
            label=r"$E_{w}||\Gamma^{* i}(w) - \Gamma_{linear}^i(w)||_2$",
        )
        plt.plot(
            range(1, self.n_iterations + 1),
            self.iterated_pbo_on_weights_error,
            color="grey",
            label=r"$E_{w}||\Gamma^{* i}(w) - \Gamma_{linear\_on\_weights}^i(w)||_2$",
        )
        plt.axhline(
            y=np.square(self.optimal_fixed_point - self.fixed_point).mean(),
            color="g",
            label=r"$||\Gamma^{* w} - \Gamma_{linear}^w||_2$",
        )
        plt.axhline(
            y=self.pbo_on_weighs_fixed_point_error,
            color="grey",
            label=r"$||\Gamma^{* w} - \Gamma_{linear\_on\_weights}^w||_2$",
        )

        plt.ylabel("errors")
        plt.xlabel("iteration")
        plt.xticks(range(1, self.n_iterations + 1), labels=range(1, self.n_iterations + 1))
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
    def add_points(ax, points: np.ndarray, color: float, alpha: float, label: str) -> None:
        xdata = points[:, 0]
        ydata = points[:, 1]
        zdata = points[:, 2]
        ax.scatter3D(xdata, ydata, zdata, color=color, alpha=alpha, label=label, s=5)

    def visualize(self, pbo: BasePBO, optimal: bool) -> None:
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
        plt.title("Iterations ")
        fig.tight_layout()
        plt.show()
