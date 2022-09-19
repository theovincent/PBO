from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk

from pbo.networks.base_q import BaseQ
from pbo.networks.base_pbo import BasePBO


class LinearPBONet(hk.Module):
    def __init__(self, layer_dimension: int, initial_weight_std: float) -> None:
        super().__init__(name="LinearPBONet")
        self.layer_dimension = layer_dimension
        self.initial_weight_std = initial_weight_std

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(
            self.layer_dimension, name="linear", w_init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std)
        )(weights)

        return x


class LinearPBO(BasePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        network_key: int,
        learning_rate: dict,
        initial_weight_std: float,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q.weights_dimension, initial_weight_std)(weights)

        super().__init__(
            q=q,
            max_bellman_iterations=max_bellman_iterations,
            add_infinity=add_infinity,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )

    @partial(jax.jit, static_argnames="self")
    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        return jnp.linalg.solve(
            jnp.eye(self.q.weights_dimension) - params["LinearPBONet/linear"]["w"].T,
            params["LinearPBONet/linear"]["b"].T,
        ).T

    def contracting_factor(self) -> float:
        return jnp.linalg.norm(self.params["LinearPBONet/linear"]["w"], ord=1)


class MaxLinearPBONet(hk.Module):
    def __init__(self, n_actions: int, layer_dimension: int, initial_weight_std: float) -> None:
        super().__init__(name="MaxLinearPBONet")
        self.n_actions = n_actions
        self.layer_dimension = layer_dimension
        self.initial_weight_std = initial_weight_std

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = hk.MaxPool(window_shape=self.n_actions, strides=self.n_actions, padding="VALID", channel_axis=0)(weights)
        x = hk.Linear(
            self.layer_dimension, name="linear", w_init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std)
        )(x)

        return x


class MaxLinearPBO(BasePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        network_key: int,
        learning_rate: dict,
        n_actions: int,
        initial_weight_std: float,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return MaxLinearPBONet(n_actions, q.weights_dimension, initial_weight_std)(weights)

        super().__init__(
            q=q,
            max_bellman_iterations=max_bellman_iterations,
            add_infinity=False,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )


class CustomLinearPBONet(hk.Module):
    def __init__(self, initial_weight_std: float) -> None:
        super().__init__(name="CustomLinearPBONet")
        self.initial_weight_std = initial_weight_std

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        customs = weights[:, 0] - weights[:, 1] ** 2 / (weights[:, 2] + 1e-32)

        slope = hk.get_parameter(
            "slope", (1, 3), weights.dtype, init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std)
        )
        bias = hk.get_parameter(
            "bias", (1, 3), weights.dtype, init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std)
        )

        return customs.reshape((-1, 1)) @ slope + bias


class CustomLinearPBO(BasePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        network_key: int,
        learning_rate: dict,
        initial_weight_std: float,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return CustomLinearPBONet(initial_weight_std)(weights)

        super().__init__(
            q=q,
            max_bellman_iterations=max_bellman_iterations,
            add_infinity=False,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )


class LinearMaxLinearPBONet(hk.Module):
    def __init__(self, layer_dimension: int, initial_weight_std: float) -> None:
        super().__init__(name="LinearPBONet")
        self.layer_dimension = layer_dimension
        self.initial_weight_std = initial_weight_std

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(
            2 * self.layer_dimension,
            name="linear1",
            w_init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std),
        )(weights)
        x = jax.nn.relu(x)
        x = hk.Linear(
            self.layer_dimension,
            name="linear2",
            w_init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std),
        )(x)
        x = hk.MaxPool(window_shape=2, strides=2, padding="VALID", channel_axis=0)(x)
        x = hk.Linear(
            self.layer_dimension,
            name="linear3",
            w_init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std),
        )(x)
        x = jax.nn.relu(x)
        x = hk.Linear(
            self.layer_dimension,
            name="linear4",
            w_init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std),
        )(x)

        return x


class LinearMaxLinearPBO(BasePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        network_key: int,
        learning_rate: dict,
        initial_weight_std: float,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearMaxLinearPBONet(q.weights_dimension, initial_weight_std)(weights)

        super().__init__(
            q=q,
            max_bellman_iterations=max_bellman_iterations,
            add_infinity=False,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )
