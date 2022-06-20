from functools import partial

import haiku as hk
import jax.numpy as jnp
import jax
import optax

from pbo.networks.q import BaseQ


class BasePBO:
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
    ) -> None:
        self.q = q
        self.max_bellman_iterations = max_bellman_iterations
        self.add_infinity = add_infinity
        self.importance_iteration = jnp.array(importance_iteration)
        assert len(importance_iteration) == (
            self.max_bellman_iterations + 1
        ), f"The number of importance iteration: {len(importance_iteration)} has to be the same as the number of iteration + 1: {self.max_bellman_iterations + 1}"

    def __call__(self, params: hk.Params, weights: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, batch_weights: jnp.ndarray, batch_samples: dict) -> jnp.ndarray:
        return jax.vmap(
            lambda weights: batch_samples["reward"]
            + self.q.max_value(self.q.to_params(weights), batch_samples["next_state"])
        )(batch_weights)

    @partial(jax.jit, static_argnames=("self", "ord"))
    def loss(self, pbo_params: hk.Params, batch_weights: jnp.ndarray, sample: dict, ord: int = 2) -> float:
        loss = 0
        batch_iterated_weights = batch_weights

        for idx_iteration in jnp.arange(self.max_bellman_iterations):
            batch_targets = self.compute_target(batch_iterated_weights, sample)
            batch_targets = jax.lax.stop_gradient(batch_targets)

            batch_iterated_weights = self(pbo_params, batch_iterated_weights)

            q_values = jax.vmap(lambda weights: self.q(self.q.to_params(weights), sample["state"], sample["action"]))(
                batch_iterated_weights
            )

            if ord == 1:
                loss += self.importance_iteration[idx_iteration] * jnp.abs(q_values - batch_targets).mean()
            else:
                loss += self.importance_iteration[idx_iteration] * jnp.linalg.norm(q_values - batch_targets)

        loss /= self.max_bellman_iterations

        if self.add_infinity:
            fixed_point = self.fixed_point(pbo_params).reshape((1, -1))
            batch_targets = self.compute_target(fixed_point, sample)
            batch_targets = jax.lax.stop_gradient(batch_targets)

            q_values = jax.vmap(lambda weights: self.q(self.q.to_params(weights), sample["state"], sample["action"]))(
                fixed_point
            )

            if ord == 1:
                loss += self.importance_iteration[-1] * jnp.abs(q_values - batch_targets).mean()
            else:
                loss += self.importance_iteration[-1] * jnp.linalg.norm(q_values - batch_targets)

        return loss

    @partial(jax.jit, static_argnames=("self", "n_discrete_states", "n_discrete_actions", "ord"))
    def loss_mesh(
        self,
        pbo_params: hk.Params,
        batch_weights: jnp.ndarray,
        sample: dict,
        n_discrete_states: int,
        n_discrete_actions: int,
        ord: int = 2,
    ) -> jnp.ndarray:
        loss_mesh = jnp.zeros((n_discrete_states, n_discrete_actions))
        batch_iterated_weights = batch_weights

        for idx_iteration in jnp.arange(self.max_bellman_iterations):
            batch_targets = self.compute_target(batch_iterated_weights, sample).reshape(
                (-1, n_discrete_states, n_discrete_actions)
            )
            batch_iterated_weights = self(pbo_params, batch_iterated_weights)

            q_values = jax.vmap(lambda weights: self.q(self.q.to_params(weights), sample["state"], sample["action"]))(
                batch_iterated_weights
            ).reshape((-1, n_discrete_states, n_discrete_actions))

            if ord == 1:
                loss_mesh = loss_mesh.at[:].add(
                    self.importance_iteration[idx_iteration] * jnp.abs(q_values - batch_targets).mean(axis=0)
                )
            else:
                loss_mesh = loss_mesh.at[:].add(
                    self.importance_iteration[idx_iteration] * jnp.linalg.norm(q_values - batch_targets, axis=0)
                )

        loss_mesh /= self.max_bellman_iterations

        if self.add_infinity:
            fixed_point = self.fixed_point(pbo_params).reshape((1, -1))
            batch_targets = self.compute_target(fixed_point, sample).reshape(
                (-1, n_discrete_states, n_discrete_actions)
            )

            q_values = jax.vmap(lambda weights: self.q(self.q.to_params(weights), sample["state"], sample["action"]))(
                fixed_point
            ).reshape((-1, n_discrete_states, n_discrete_actions))

            if ord == 1:
                loss_mesh = loss_mesh.at[:].add(
                    self.importance_iteration[-1] * jnp.abs(q_values - batch_targets).mean(axis=0)
                )
            else:
                loss_mesh = loss_mesh.at[:].add(
                    self.importance_iteration[-1] * jnp.linalg.norm(q_values - batch_targets, axis=0)
                )

        return loss_mesh

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        raise NotImplementedError


class BaseOptimalPBO(BasePBO):
    def __init__(self, q: BaseQ, max_bellman_iterations: int, add_infinity: bool, importance_iteration: list) -> None:
        super().__init__(q, max_bellman_iterations, add_infinity, importance_iteration)


class Optimal3DPBO(BaseOptimalPBO):
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


class OptimalTablePBO(BaseOptimalPBO):
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
