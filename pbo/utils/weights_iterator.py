import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import jax.numpy as jnp
import haiku as hk

from pbo.environment.linear_quadratic import LinearQuadraticEnv
from pbo.networks.pbo import BasePBO, OptimalPBO


class WeightsIterator:
    def __init__(
        self, env: LinearQuadraticEnv, pbo: BasePBO, n_iterations: int, weights: jnp.ndarray, sleeping_time: float
    ) -> None:
        self.optimal_weights = [
            env.Q[0, 0] + env.A[0, 0] ** 2 * env.P[0, 0],
            env.S[0, 0] + env.A[0, 0] * env.B[0, 0] * env.P[0, 0],
            env.R[0, 0] + env.B[0, 0] ** 2 * env.P[0, 0],
        ]

        self.pbo_apply = pbo.network.apply
        self.optimal_pbo = OptimalPBO(env)

        self.n_iterations = n_iterations
        self.iterated_weights = np.zeros((n_iterations, weights.shape[0]))
        self.iterated_weights[0] = weights

        self.optimal_iterated_weights = np.zeros((n_iterations, weights.shape[0]))
        self.optimal_iterated_weights[0] = weights

        for iteration in range(1, self.n_iterations):
            self.optimal_iterated_weights[iteration] = self.optimal_pbo(self.optimal_iterated_weights[iteration - 1])

        self.sleeping_time = sleeping_time

    def iterate_on_params(self, params: hk.Params, fixed_point: jnp.ndarray) -> None:
        for iteration in range(1, self.n_iterations):
            self.iterated_weights[iteration] = self.pbo_apply(params, self.iterated_weights[iteration - 1])

        self.fixed_point = fixed_point

    def show(self, title: str = "") -> None:
        clear_output(wait=True)
        fig, axes = plt.subplots(3, 1)
        labels = ["k", "i", " m"]

        for idx_ax, ax in enumerate(axes):
            ax.axhline(y=self.optimal_weights[idx_ax], color="black", label="Optimal weights")
            ax.axhline(y=self.fixed_point[idx_ax], color="grey", label="Fixed point", linestyle="--")
            ax.scatter(
                np.arange(self.n_iterations),
                self.optimal_iterated_weights[:, idx_ax],
                color="black",
                label="Optimally iterated weights",
            )
            ax.scatter(
                np.arange(self.n_iterations), self.iterated_weights[:, idx_ax], color="r", label="Iterated weights"
            )

            ax.set_ylim(
                self.optimal_weights[idx_ax] - 10 * self.optimal_weights[idx_ax],
                self.optimal_weights[idx_ax] + 10 * self.optimal_weights[idx_ax],
            )
            ax.set_ylabel(labels[idx_ax])
            if idx_ax != 2:
                ax.set_xticks([])
                ax.set_xticklabels([])

        if title != "":
            axes[0].set_title(title)
        axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        axes[2].set_xlabel("iterations")

        fig.tight_layout()
        fig.canvas.draw()
        plt.show()
        time.sleep(self.sleeping_time)
