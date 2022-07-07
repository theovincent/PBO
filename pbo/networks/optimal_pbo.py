import jax.numpy as jnp

from pbo.networks.base_q import BaseQ
from pbo.networks.base_pbo import BasePBO


class OptimalPBO(BasePBO):
    def __init__(self, q: BaseQ, max_bellman_iterations: int, add_infinity: bool) -> None:
        super().__init__(q, max_bellman_iterations, add_infinity)


class OptimalLQRPBO(OptimalPBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        A: float,
        B: float,
        Q: float,
        R: float,
        S: float,
        P: float,
    ) -> None:
        super().__init__(q, max_bellman_iterations, add_infinity)
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.S = S
        self.P = P

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        # a batch of weights comes with shape (b_s, 3)
        # estimated_p is of shape (b_s)
        estimated_p = weights.T[0] - weights.T[1] ** 2 / weights.T[2]

        return jnp.array(
            [
                self.Q + self.A**2 * estimated_p,
                self.S + self.A * self.B * estimated_p,
                self.R + self.B**2 * estimated_p,
            ]
        ).T

    def fixed_point(self) -> jnp.ndarray:
        return jnp.array(
            [
                self.Q + self.A**2 * self.P,
                self.S + self.A * self.B * self.P,
                self.R + self.B**2 * self.P,
            ]
        )
