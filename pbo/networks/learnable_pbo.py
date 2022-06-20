from functools import partial

import jax.numpy as jnp
import jax
import haiku as hk
import optax

from pbo.networks.q import BaseQ
from pbo.networks.base_pbo import BasePBO
from pbo.networks.optimal_pbo import OptimalPBO


class LearnablePBO(BasePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
        network: hk.Module,
        network_key: int,
        learning_rate: float,
    ) -> None:
        super().__init__(q, max_bellman_iterations, add_infinity, importance_iteration)

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, weights=jnp.zeros((q.weights_dimension)))

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

        learning_rate_schedule = optax.linear_schedule(
            learning_rate["first"], learning_rate["last"], learning_rate["duration"]
        )
        self.optimizer = optax.adam(learning_rate_schedule)
        self.optimizer_state = self.optimizer.init(self.params)

    @partial(jax.jit, static_argnames="self")
    def __call__(self, params: hk.Params, weights: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, weights)

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: hk.Params, optimizer_state: tuple, batch_weights: jnp.ndarray, batch_samples: jnp.ndarray
    ) -> tuple:
        loss, grad_loss = self.loss_and_grad(params, batch_weights, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss


class LinearPBONet(hk.Module):
    def __init__(self, layer_dimension: int) -> None:
        super().__init__(name="LinearPBONet")
        self.layer_dimension = layer_dimension

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.layer_dimension, name="linear")(weights)

        return x


class LinearPBO(LearnablePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
        network_key: int,
        learning_rate: float,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q.weights_dimension)(weights)

        super().__init__(
            q, max_bellman_iterations, add_infinity, importance_iteration, network, network_key, learning_rate
        )

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        return jnp.linalg.solve(
            jnp.eye(self.q.weights_dimension) - params["LinearPBONet/linear"]["w"].T,
            params["LinearPBONet/linear"]["b"].T,
        ).T

    def contracting_factor(self) -> float:
        return jnp.linalg.norm(self.params["LinearPBONet/linear"]["w"], ord=1)


class LearnableOnWeightsPBO(LearnablePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
        network: hk.Module,
        network_key: int,
        learning_rate: float,
        pbo_optimal: OptimalPBO,
    ) -> None:
        super().__init__(
            q, max_bellman_iterations, add_infinity, importance_iteration, network, network_key, learning_rate
        )
        self.loss_and_grad_on_weights = jax.jit(jax.value_and_grad(self.loss_on_weights))
        self.pbo_optimal = pbo_optimal

    @partial(jax.jit, static_argnames=("self", "ord"))
    def loss_on_weights(self, pbo_params: hk.Params, batch_weights: jnp.ndarray, ord: int = 2) -> jnp.ndarray:
        loss = 0
        batch_weights_iterated = batch_weights
        batch_weights_optimal_iterated = batch_weights

        for idx_iteration in jnp.arange(self.max_bellman_iterations):
            batch_weights_iterated = self(pbo_params, batch_weights_iterated)
            batch_weights_optimal_iterated = self.pbo_optimal(batch_weights_optimal_iterated)
            batch_weights_optimal_iterated = jax.lax.stop_gradient(batch_weights_optimal_iterated)

            if ord == 1:
                loss += (
                    self.importance_iteration[idx_iteration]
                    * jnp.abs(batch_weights_optimal_iterated - batch_weights_iterated).mean()
                )
            else:
                loss += self.importance_iteration[idx_iteration] * jnp.linalg.norm(
                    batch_weights_optimal_iterated - batch_weights_iterated
                )

        loss /= self.max_bellman_iterations

        if self.add_infinity:
            batch_fixed_point = self.fixed_point(pbo_params)
            if ord == 1:
                loss += (
                    self.importance_iteration[-1] * jnp.abs(batch_fixed_point - self.pbo_optimal.fixed_point()).mean()
                )
            else:
                loss += self.importance_iteration[-1] * jnp.linalg.norm(
                    batch_fixed_point - self.pbo_optimal.fixed_point()
                )

        return loss

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch_on_weights(self, params: hk.Params, optimizer_state: tuple, batch_weights: jnp.ndarray) -> tuple:
        loss, grad_loss = self.loss_and_grad_on_weights(params, batch_weights)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss, grad_loss


class LinearPBOOnWeights(LearnableOnWeightsPBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
        network_key: int,
        learning_rate: float,
        pbo_optimal: OptimalPBO,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q.weights_dimension)(weights)

        super().__init__(
            q,
            max_bellman_iterations,
            add_infinity,
            importance_iteration,
            network,
            network_key,
            learning_rate,
            pbo_optimal,
        )

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        return jnp.linalg.solve(
            jnp.eye(self.q.weights_dimension) - params["LinearPBONet/linear"]["w"].T,
            params["LinearPBONet/linear"]["b"].T,
        ).T

    def contracting_factor(self) -> float:
        return jnp.linalg.norm(self.params["LinearPBONet/linear"]["w"], ord=1)
