from functools import partial

import jax.numpy as jnp
import jax
import haiku as hk
import optax

from pbo.networks.base_q import BaseQ
from pbo.networks.base_pbo import BasePBO
from pbo.networks.optimal_pbo import OptimalPBO


class LearnablePBO(BasePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        network: hk.Module,
        network_key: int,
        learning_rate: dict,
    ) -> None:
        super().__init__(q, max_bellman_iterations, add_infinity)

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, weights=jnp.zeros((1, q.weights_dimension)))

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

        self.learning_rate_schedule = optax.linear_schedule(
            learning_rate["first"], learning_rate["last"], learning_rate["duration"]
        )
        self.optimizer = optax.adam(self.learning_rate_schedule)
        self.optimizer_state = self.optimizer.init(self.params)

    @partial(jax.jit, static_argnames="self")
    def __call__(self, params: hk.Params, weights: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, weights)

    def reset_optimizer(self) -> None:
        self.optimizer = optax.adam(self.learning_rate_schedule)
        self.optimizer_state = self.optimizer.init(self.params)

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: hk.Params,
        params_target: hk.Params,
        optimizer_state: tuple,
        batch_weights: jnp.ndarray,
        batch_samples: jnp.ndarray,
        importance_iteration: jnp.ndarray,
    ) -> tuple:
        loss, grad_loss = self.loss_and_grad(params, params_target, batch_weights, batch_samples, importance_iteration)
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
        network_key: int,
        learning_rate: dict,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q.weights_dimension)(weights)

        super().__init__(q, max_bellman_iterations, add_infinity, network, network_key, learning_rate)

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
        network: hk.Module,
        network_key: int,
        learning_rate: dict,
        pbo_optimal: OptimalPBO,
    ) -> None:
        super().__init__(q, max_bellman_iterations, add_infinity, network, network_key, learning_rate)
        self.loss_and_grad_on_weights = jax.jit(jax.value_and_grad(self.loss_on_weights))
        self.pbo_optimal = pbo_optimal

    @partial(jax.jit, static_argnames=("self", "ord"))
    def loss_on_weights(
        self, pbo_params: hk.Params, batch_weights: jnp.ndarray, importance_iteration: jnp.ndarray, ord: int = 2
    ) -> jnp.ndarray:
        loss = 0
        batch_weights_iterated = batch_weights
        batch_weights_optimal_iterated = batch_weights

        for idx_iteration in jnp.arange(self.max_bellman_iterations):
            batch_weights_iterated = jax.lax.stop_gradient(batch_weights_iterated)
            batch_weights_iterated = self(pbo_params, batch_weights_iterated)
            batch_weights_optimal_iterated = self.pbo_optimal(batch_weights_optimal_iterated)
            batch_weights_optimal_iterated = jax.lax.stop_gradient(batch_weights_optimal_iterated)

            if ord == 1:
                loss += (
                    importance_iteration[idx_iteration]
                    * jnp.abs(batch_weights_optimal_iterated - batch_weights_iterated).mean()
                )
            else:
                loss += (
                    importance_iteration[idx_iteration]
                    * jnp.square(batch_weights_optimal_iterated - batch_weights_iterated).mean()
                )

        if self.add_infinity:
            loss /= self.max_bellman_iterations

            batch_fixed_point = self.fixed_point(pbo_params)
            if ord == 1:
                loss += importance_iteration[-1] * jnp.abs(batch_fixed_point - self.pbo_optimal.fixed_point()).mean()
            else:
                loss += importance_iteration[-1] * jnp.square(batch_fixed_point - self.pbo_optimal.fixed_point()).mean()

        return loss

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch_on_weights(
        self,
        params: hk.Params,
        optimizer_state: jnp.ndarray,
        batch_weights: jnp.ndarray,
        importance_iteration: jnp.ndarray,
    ) -> tuple:
        loss, grad_loss = self.loss_and_grad_on_weights(params, batch_weights, importance_iteration)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss, grad_loss


class LinearPBOOnWeights(LearnableOnWeightsPBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        network_key: int,
        learning_rate: dict,
        pbo_optimal: OptimalPBO,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q.weights_dimension)(weights)

        super().__init__(
            q,
            max_bellman_iterations,
            add_infinity,
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


class TabularPBONet(hk.Module):
    def __init__(self, n_actions: int, layer_dimension: int) -> None:
        super().__init__(name="TabularPBONet")
        self.n_actions = n_actions
        self.layer_dimension = layer_dimension

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = hk.MaxPool(window_shape=self.n_actions, strides=self.n_actions, padding="VALID", channel_axis=0)(weights)
        x = hk.Linear(self.layer_dimension, name="linear")(x)

        return x


class TabularPBO(LearnablePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        network_key: int,
        learning_rate: dict,
        n_actions: int,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return TabularPBONet(n_actions, q.weights_dimension)(weights)

        super().__init__(q, max_bellman_iterations, add_infinity, network, network_key, learning_rate)
