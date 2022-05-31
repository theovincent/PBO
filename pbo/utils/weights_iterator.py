import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import jax.numpy as jnp
import haiku as hk

from pbo.networks.pbo import OptimalPBO, OptimalLinearPBO


class WeightsIterator:
    def __init__(
        self,
        pbo_apply,
        pbo_optimal: OptimalPBO,
        pbo_optimal_linear: OptimalLinearPBO,
        weights: jnp.ndarray,
        n_iterations: int,
        A: float,
        B: float,
        Q: float,
        R: float,
        S: float,
        P: float,
        sleeping_time: float,
    ) -> None:
        self.pbo_apply = pbo_apply
        self.n_iterations = n_iterations

        self.optimal_weights = jnp.array(
            [
                Q + A**2 * P,
                S + A * B * P,
                R + B**2 * P,
            ]
        )

        self.optimal_linear_weights = pbo_optimal_linear.fixed_point()
        self.optimal_weights_error = np.linalg.norm(self.optimal_weights - self.optimal_linear_weights)

        self.iterated_weights = np.zeros((n_iterations, weights.shape[0], weights.shape[1]))
        self.iterated_weights[0] = weights

        self.iterated_weights_optimal = np.zeros((n_iterations, weights.shape[0], weights.shape[1]))
        self.iterated_weights_optimal[0] = weights
        self.iterated_weights_optimal_linear = np.zeros((n_iterations, weights.shape[0], weights.shape[1]))
        self.iterated_weights_optimal_linear[0] = weights

        for iteration in range(1, self.n_iterations):
            self.iterated_weights_optimal[iteration] = pbo_optimal(self.iterated_weights_optimal[iteration - 1])
            self.iterated_weights_optimal_linear[iteration] = pbo_optimal_linear.network.apply(
                pbo_optimal_linear.params, self.iterated_weights_optimal_linear[iteration - 1]
            )

        self.n_weigths = weights.shape[0]
        self.iterated_optimal_error = (
            np.linalg.norm(self.iterated_weights_optimal - self.iterated_weights_optimal_linear, axis=(1, 2))
            / self.n_weigths
        )
        self.iterated_optimal_error_std = np.linalg.norm(
            self.iterated_weights_optimal - self.iterated_weights_optimal_linear, axis=2
        ).std(axis=1)

        self.sleeping_time = sleeping_time

    def iterate_on_params(self, params: hk.Params, fixed_point: jnp.ndarray) -> None:
        for iteration in range(1, self.n_iterations):
            self.iterated_weights[iteration] = self.pbo_apply(params, self.iterated_weights[iteration - 1])

        self.fixed_point = fixed_point

    def show(self, title: str = "") -> None:
        clear_output(wait=True)

        # Plot the iterations on the first weight
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 5))
        labels = ["k", "i", " m"]

        for idx_ax, ax in enumerate(axes):
            ax.axhline(y=self.optimal_weights[idx_ax], color="black", label="Optimal weights")
            ax.axhline(
                y=self.optimal_linear_weights[idx_ax], color="grey", label="Optimal linear weights", linestyle="--"
            )
            ax.axhline(y=self.fixed_point[idx_ax], color="g", label="fixed point", linestyle="--")
            ax.scatter(
                range(self.n_iterations),
                self.iterated_weights_optimal[:, 0, idx_ax],
                color="black",
                label="Optimally iterated weights",
            )
            ax.scatter(
                range(self.n_iterations),
                self.iterated_weights_optimal_linear[:, 0, idx_ax],
                color="grey",
                label="Optimally linear iterated weights",
                marker="x",
            )
            ax.scatter(
                range(self.n_iterations),
                self.iterated_weights[:, 0, idx_ax],
                color="g",
                label="Iterated weights",
                marker="x",
            )

            ax.set_ylim(
                self.optimal_weights[idx_ax] - 2 * abs(self.optimal_weights[idx_ax]),
                self.optimal_weights[idx_ax] + 2 * abs(self.optimal_weights[idx_ax]),
            )
            ax.set_ylabel(labels[idx_ax])
            if idx_ax != 2:
                ax.set_xticks([])
                ax.set_xticklabels([])
            else:
                ax.set_xticks(range(self.n_iterations))
                ax.set_xticklabels(range(self.n_iterations))

        if title != "":
            axes[0].set_title(title)
        axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        axes[2].set_xlabel("iterations")

        fig.tight_layout()
        fig.canvas.draw()

        # Plot the errors on all weights
        plt.figure(figsize=(7, 3))

        iterated_error = (
            np.linalg.norm(self.iterated_weights_optimal - self.iterated_weights, axis=(1, 2)) / self.n_weigths
        )
        iterated_error_std = np.linalg.norm(self.iterated_weights_optimal - self.iterated_weights, axis=2).std(axis=1)

        plt.bar(
            range(self.n_iterations),
            iterated_error,
            yerr=iterated_error_std,
            color="g",
            ecolor="darkgreen",
            label="||T*(w) - T_phi(w)||",
        )
        plt.bar(
            range(self.n_iterations),
            self.iterated_optimal_error,
            yerr=self.iterated_optimal_error_std,
            color="grey",
            ecolor="dimgrey",
            label="||T*(w) - T*_linear(w)||",
            alpha=0.7,
        )
        plt.axhline(
            y=np.linalg.norm(self.optimal_weights - self.fixed_point), color="g", label="||opt_w - fixed_point||"
        )
        plt.axhline(y=self.optimal_weights_error, color="grey", label="||opt_w - fixed_point_linear||")

        plt.ylabel("errors")
        plt.xlabel("iteration")
        not_nan = tuple(~jnp.isnan(iterated_error + iterated_error_std))
        plt.ylim(0, iterated_error[not_nan].max() + iterated_error_std[not_nan].max())
        plt.title(
            f"PBO error: {str(jnp.round(iterated_error.sum(), 2))}, Optimal linear error: {str(jnp.round(self.iterated_optimal_error.sum(), 2))}"
        )
        plt.legend()
        plt.show()

        time.sleep(self.sleeping_time)
