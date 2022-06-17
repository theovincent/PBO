import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import jax.numpy as jnp
import haiku as hk

from pbo.networks.pbo import BaseOptimalPBO


class OperatorViewer:
    def __init__(
        self,
        pbo_apply,
        pbo_optimal: BaseOptimalPBO,
        weights: jnp.ndarray,
        weights_range: float,
        sleeping_time: float,
    ) -> None:
        self.pbo_apply = pbo_apply
        self.parameters = jnp.linspace(-weights_range, weights_range, 2).reshape((-1, 1))
        self.weights = weights

        self.optimal_iterations = pbo_optimal(self.parameters)
        self.optimal_iterations_on_weights = pbo_optimal(weights)

        self.sleeping_time = sleeping_time

    def update_pbo_iterations(self, params: hk.Params) -> None:
        self.pbo_iterations = self.pbo_apply(params, self.parameters)
        self.iterations_on_weights = self.pbo_apply(params, self.weights)

    def show(self, title: str = "") -> None:
        clear_output(wait=True)

        _, ax = plt.subplots(figsize=(5, 5))

        ax.plot(self.parameters, self.parameters, color="brown", label="y=x")
        ax.plot(self.parameters, self.optimal_iterations, color="black", label="Optimal PBO")
        ax.plot(self.parameters, self.pbo_iterations, color="g", label="PBO")

        ax.scatter(self.weights, self.optimal_iterations_on_weights, color="black", label="Optimal iteration")
        ax.scatter(self.weights, self.iterations_on_weights, color="g", marker="x", label="iteration")

        ax.set_xlabel("parameter K")
        ax.legend()
        ax.set_aspect("equal", "box")
        if title != "":
            ax.set_title(title)
        plt.show()

        time.sleep(self.sleeping_time)
