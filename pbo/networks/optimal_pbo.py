import jax.numpy as jnp
import jax

from pbo.networks.q import BaseQ
from pbo.networks.base_pbo import BasePBO


class OptimalPBO(BasePBO):
    def __init__(self, q: BaseQ, max_bellman_iterations: int, add_infinity: bool, importance_iteration: list) -> None:
        super().__init__(q, max_bellman_iterations, add_infinity, importance_iteration)


class Optimal3DPBO(OptimalPBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
        A: float,
        B: float,
        Q: float,
        R: float,
        S: float,
        P: float,
    ) -> None:
        super().__init__(q, max_bellman_iterations, add_infinity, importance_iteration)
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


class OptimalTablePBO(OptimalPBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
        optimal_bellman_operator,
        optimal_q: jnp.ndarray,
    ) -> None:
        super().__init__(q, max_bellman_iterations, add_infinity, importance_iteration)
        self.optimal_bellman_operator = optimal_bellman_operator
        self.optimal_q = optimal_q

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda weights_: self.optimal_bellman_operator(self.q.to_params(weights_)["TableQNet"]["table"]).flatten()
        )(weights)

    def fixed_point(self) -> jnp.ndarray:
        return self.optimal_q
