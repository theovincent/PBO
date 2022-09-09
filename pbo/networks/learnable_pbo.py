from functools import partial
from re import L

import jax.numpy as jnp
import jax
import haiku as hk
import optax

from pbo.networks.base_q import BaseQ
from pbo.networks.base_pbo import BasePBO


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

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss), static_argnames="ord")

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

    @partial(jax.jit, static_argnames=("self", "ord"))
    def learn_on_batch(
        self,
        params: hk.Params,
        params_target: hk.Params,
        optimizer_state: tuple,
        batch_weights: jnp.ndarray,
        batch_samples: jnp.ndarray,
        importance_iteration: jnp.ndarray,
        ord: int = 2,
    ) -> tuple:
        loss, grad_loss = self.loss_and_grad(
            params, params_target, batch_weights, batch_samples, importance_iteration, ord
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss


class LinearPBONet(hk.Module):
    def __init__(self, layer_dimension: int, initial_weight_std: float, initial_bias_std: float) -> None:
        super().__init__(name="LinearPBONet")
        self.layer_dimension = layer_dimension
        self.initial_weight_std = initial_weight_std
        self.initial_bias_std = initial_bias_std

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(
            self.layer_dimension,
            name="linear",
            w_init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std),
            b_init=hk.initializers.TruncatedNormal(stddev=self.initial_bias_std),
        )(weights)

        return x


class LinearPBO(LearnablePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        network_key: int,
        learning_rate: dict,
        initial_weight_std: float,
        initial_bias_std: float,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q.weights_dimension, initial_weight_std, initial_bias_std)(weights)

        super().__init__(q, max_bellman_iterations, add_infinity, network, network_key, learning_rate)

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        return jnp.linalg.solve(
            jnp.eye(self.q.weights_dimension) - params["LinearPBONet/linear"]["w"].T,
            params["LinearPBONet/linear"]["b"].T,
        ).T

    def contracting_factor(self) -> float:
        return jnp.linalg.norm(self.params["LinearPBONet/linear"]["w"], ord=1)


class MaxLinearPBONet(hk.Module):
    def __init__(
        self, n_actions: int, layer_dimension: int, initial_weight_std: float, initial_bias_std: float
    ) -> None:
        super().__init__(name="MaxLinearPBONet")
        self.n_actions = n_actions
        self.layer_dimension = layer_dimension
        self.initial_weight_std = initial_weight_std
        self.initial_bias_std = initial_bias_std

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = hk.MaxPool(window_shape=self.n_actions, strides=self.n_actions, padding="VALID", channel_axis=0)(weights)
        x = hk.Linear(
            self.layer_dimension,
            name="linear",
            w_init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std),
            b_init=hk.initializers.TruncatedNormal(stddev=self.initial_bias_std),
        )(x)

        return x


class MaxLinearPBO(LearnablePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        network_key: int,
        learning_rate: dict,
        n_actions: int,
        initial_weight_std: float,
        initial_bias_std: float,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return MaxLinearPBONet(n_actions, q.weights_dimension, initial_weight_std, initial_bias_std)(weights)

        super().__init__(q, max_bellman_iterations, False, network, network_key, learning_rate)


class CustomLinearPBONet(hk.Module):
    def __init__(self, initial_weight_std: float, initial_bias_std: float) -> None:
        super().__init__(name="CustomLinearPBONet")
        self.initial_weight_std = initial_weight_std
        self.initial_bias_std = initial_bias_std

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        customs = weights[:, 0] - weights[:, 1] ** 2 / (weights[:, 2] + 1e-32)

        slope = hk.get_parameter(
            "slope", (1, 3), weights.dtype, init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std)
        )
        bias = hk.get_parameter(
            "bias", (1, 3), weights.dtype, init=hk.initializers.TruncatedNormal(stddev=self.initial_bias_std)
        )

        return customs.reshape((-1, 1)) @ slope + bias


class CustomLinearPBO(LearnablePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        network_key: int,
        learning_rate: dict,
        initial_weight_std: float,
        initial_bias_std: float,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return CustomLinearPBONet(initial_weight_std, initial_bias_std)(weights)

        super().__init__(q, max_bellman_iterations, False, network, network_key, learning_rate)


class LinearMaxLinearPBONet(hk.Module):
    def __init__(self, layer_dimension: int, initial_weight_std: float, initial_bias_std: float) -> None:
        super().__init__(name="LinearPBONet")
        self.layer_dimension = layer_dimension
        self.initial_weight_std = initial_weight_std
        self.initial_bias_std = initial_bias_std

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(
            self.layer_dimension,
            name="linear1",
            w_init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std),
            b_init=hk.initializers.TruncatedNormal(stddev=self.initial_bias_std),
        )(weights)
        x = hk.MaxPool(window_shape=2, strides=2, padding="VALID", channel_axis=0)(x)
        x = hk.Linear(
            self.layer_dimension,
            name="linear2",
            w_init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std),
            b_init=hk.initializers.TruncatedNormal(stddev=self.initial_bias_std),
        )(x)

        return x


class LinearMaxLinearPBO(LearnablePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        network_key: int,
        learning_rate: dict,
        initial_weight_std: float,
        initial_bias_std: float,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearMaxLinearPBONet(q.weights_dimension, initial_weight_std, initial_bias_std)(weights)

        super().__init__(q, max_bellman_iterations, False, network, network_key, learning_rate)
