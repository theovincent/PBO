from functools import partial

import haiku as hk
import optax
import jax.numpy as jnp
import jax

from pbo.networks.base_q import BaseQ


class BasePBO:
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        add_infinity: bool,
        network: hk.Module,
        network_key: int,
        learning_rate: dict,
    ) -> None:
        self.q = q
        self.max_bellman_iterations = max_bellman_iterations
        self.add_infinity = add_infinity
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

    @partial(jax.jit, static_argnames=("self", "ord"))
    def loss_term(
        self,
        iterated_weights: jnp.ndarray,
        iterated_weights_target: jnp.ndarray,
        samples: dict,
        importance: float,
        ord: int = 2,
    ) -> float:
        predicted_values = jax.vmap(
            lambda weights: self.q(self.q.to_params(weights), samples["state"], samples["action"])
        )(iterated_weights)

        targets_values = self.q.compute_target(iterated_weights_target, samples)

        if ord == 1:
            return importance * jnp.abs(predicted_values - targets_values).mean()
        else:
            return importance * jnp.square(predicted_values - targets_values).mean()

    @partial(jax.jit, static_argnames=("self", "ord"))
    def loss(
        self,
        pbo_params: hk.Params,
        pbo_params_target: hk.Params,
        batch_weights: jnp.ndarray,
        samples: dict,
        importance_iteration: jnp.ndarray,
        ord: int,
    ) -> float:
        iterated_weights = self(pbo_params, batch_weights)
        iterated_weights_target = batch_weights

        loss = self.loss_term(iterated_weights, iterated_weights_target, samples, importance_iteration[0], ord)

        for idx_iteration in jnp.arange(1, self.max_bellman_iterations):
            # To limit the back propagation to one iteration
            # iterated_weights = jax.lax.stop_gradient(batch_iterated_weights)
            iterated_weights = self(pbo_params, iterated_weights)
            # To backpropagate over the target
            # iterated_weights_target = self(pbo_params, iterated_weights_target)
            iterated_weights_target = self(pbo_params_target, iterated_weights_target)

            loss += self.loss_term(
                iterated_weights, iterated_weights_target, samples, importance_iteration[idx_iteration], ord
            )

        loss /= importance_iteration.sum()

        if self.add_infinity:
            fixed_point = self.fixed_point(pbo_params).reshape((1, -1))

            loss += self.loss_term(fixed_point, fixed_point, samples, importance_iteration[-1], ord)

        return loss

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

    def reset_optimizer(self) -> None:
        self.optimizer = optax.adam(self.learning_rate_schedule)
        self.optimizer_state = self.optimizer.init(self.params)

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        raise NotImplementedError

    def contracting_factor(self) -> float:
        raise NotImplementedError
