from functools import partial

import haiku as hk
import jax.numpy as jnp
import jax
import optax

from pbo.networks.q import BaseQ


class BasePBO:
    def __init__(
        self,
        network: hk.Module,
        network_key: int,
        q: BaseQ,
        learning_rate: float,
        max_bellman_iterations: int,
    ) -> None:
        self.q = q

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, weights=jnp.zeros((q.weights_dimension)))

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

        self.iteration_loss_and_grad = []
        for n_iterations in range(max_bellman_iterations):
            self.iteration_loss_and_grad.append(
                jax.jit(jax.value_and_grad(partial(self.iteration_loss, n_iterations=n_iterations)))
            )

        learning_rate_schedule = optax.linear_schedule(
            learning_rate["first"], learning_rate["last"], learning_rate["duration"]
        )
        self.optimizer = optax.adam(learning_rate_schedule)
        self.optimizer_state = self.optimizer.init(self.params)

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, batch_samples: dict, batch_weights: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda weights: batch_samples["reward"]
            + self.q.max_value(self.q.to_params(weights), batch_samples["next_state"])
        )(batch_weights)

    def loss(
        self, pbo_params: hk.Params, sample: dict, batch_weights: jnp.ndarray, batch_targets: jnp.ndarray
    ) -> jnp.ndarray:
        batch_iterated_weights = self.network.apply(pbo_params, batch_weights)

        q_values = jax.vmap(
            lambda weights: self.q.network.apply(self.q.to_params(weights), sample["state"], sample["action"])
        )(batch_iterated_weights)

        return jnp.linalg.norm(q_values - batch_targets)

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: hk.Params, optimizer_state: tuple, batch_samples: jnp.ndarray, batch_weights: jnp.ndarray
    ) -> tuple:
        batch_targets = self.compute_target(batch_samples, batch_weights)

        loss, grad_loss = self.loss_and_grad(params, batch_samples, batch_weights, batch_targets)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def iteration_loss(
        self, pbo_params: hk.Params, sample: dict, batch_weights: jnp.ndarray, n_iterations: int
    ) -> jnp.ndarray:
        batch_iterated_weights = batch_weights
        for _ in jnp.arange(0, n_iterations):
            batch_iterated_weights = self.network.apply(pbo_params, batch_iterated_weights)
        batch_targets = self.compute_target(sample, batch_iterated_weights)

        batch_iterated_again_weights = self.network.apply(pbo_params, batch_iterated_weights)

        q_values = jax.vmap(
            lambda weights: self.q.network.apply(self.q.to_params(weights), sample["state"], sample["action"])
        )(batch_iterated_again_weights)

        return jnp.linalg.norm(q_values - batch_targets)

    @partial(jax.jit, static_argnames=("self", "n_iterations"))
    def learn_iterations_on_batch(
        self,
        params: hk.Params,
        optimizer_state: tuple,
        batch_samples: jnp.ndarray,
        batch_weights: jnp.ndarray,
        n_iterations: int,
    ) -> tuple:
        loss, grad_loss = self.iteration_loss_and_grad[n_iterations](params, batch_samples, batch_weights)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def fixed_point(self) -> jnp.ndarray:
        raise NotImplementedError


class LinearPBONet(hk.Module):
    def __init__(self, layer_dimension: int) -> None:
        super().__init__(name="LinearNet")
        self.layer_dimension = layer_dimension

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.layer_dimension, name="linear")(weights)

        return x


class LinearPBO(BasePBO):
    def __init__(self, network_key: int, q: BaseQ, learning_rate: float, max_bellman_iterations: int) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q.weights_dimension)(weights)

        super(LinearPBO, self).__init__(network, network_key, q, learning_rate, max_bellman_iterations)

    def fixed_point(self) -> jnp.ndarray:
        return (
            -jnp.linalg.inv(self.params["LinearNet/linear"]["w"] - jnp.eye(self.q.weights_dimension))
            @ self.params["LinearNet/linear"]["b"]
        )

    def contracting(self) -> float:
        return jnp.linalg.norm(self.params["LinearNet/linear"]["w"], ord=1)
