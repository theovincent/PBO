from functools import partial

import haiku as hk
import jax.numpy as jnp
import jax

from pbo.networks.base_q import BaseQ


class BasePBO:
    def __init__(self, q: BaseQ, max_bellman_iterations: int, add_infinity: bool) -> None:
        self.q = q
        self.max_bellman_iterations = max_bellman_iterations
        self.add_infinity = add_infinity

    def __call__(self, params: hk.Params, weights: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self", "ord"))
    def loss(
        self,
        pbo_params: hk.Params,
        pbo_params_target: hk.Params,
        batch_weights: jnp.ndarray,
        sample: dict,
        importance_iteration: jnp.ndarray,
        ord: int = 2,
    ) -> float:
        loss = 0
        batch_iterated_weights = batch_weights
        batch_iterated_weights_target = batch_weights

        for idx_iteration in jnp.arange(self.max_bellman_iterations):
            batch_iterated_weights = jax.lax.stop_gradient(batch_iterated_weights)
            batch_targets = self.q.compute_target(batch_iterated_weights_target, sample)

            batch_iterated_weights = self(pbo_params, batch_iterated_weights)
            batch_iterated_weights_target = self(pbo_params_target, batch_iterated_weights_target)

            q_values = jax.vmap(lambda weights: self.q(self.q.to_params(weights), sample["state"], sample["action"]))(
                batch_iterated_weights
            )

            if ord == 1:
                loss += importance_iteration[idx_iteration] * jnp.abs(q_values - batch_targets).mean()
            else:
                loss += importance_iteration[idx_iteration] * jnp.square(q_values - batch_targets).mean()

        loss /= importance_iteration.sum()

        if self.add_infinity:
            fixed_point = self.fixed_point(pbo_params).reshape((1, -1))
            batch_targets = self.q.compute_target(fixed_point, sample)

            q_values = jax.vmap(lambda weights: self.q(self.q.to_params(weights), sample["state"], sample["action"]))(
                fixed_point
            )

            if ord == 1:
                loss += importance_iteration[-1] * jnp.abs(q_values - batch_targets).mean()
            else:
                loss += importance_iteration[-1] * jnp.square(q_values - batch_targets).mean()

        return loss

    @partial(jax.jit, static_argnames=("self", "n_discrete_states", "n_discrete_actions", "ord"))
    def loss_mesh(
        self,
        pbo_params: hk.Params,
        batch_weights: jnp.ndarray,
        sample: dict,
        importance_iteration: jnp.ndarray,
        n_discrete_states: int,
        n_discrete_actions: int,
        ord: int = 2,
    ) -> jnp.ndarray:
        loss_mesh = jnp.zeros((n_discrete_states, n_discrete_actions))
        batch_iterated_weights = batch_weights

        for idx_iteration in jnp.arange(self.max_bellman_iterations):
            batch_targets = self.q.compute_target(batch_iterated_weights, sample).reshape(
                (-1, n_discrete_states, n_discrete_actions)
            )
            batch_iterated_weights = self(pbo_params, batch_iterated_weights)

            q_values = jax.vmap(lambda weights: self.q(self.q.to_params(weights), sample["state"], sample["action"]))(
                batch_iterated_weights
            ).reshape((-1, n_discrete_states, n_discrete_actions))

            if ord == 1:
                loss_mesh = loss_mesh.at[:].add(
                    importance_iteration[idx_iteration] * jnp.abs(q_values - batch_targets).mean(axis=0)
                )
            else:
                loss_mesh = loss_mesh.at[:].add(
                    importance_iteration[idx_iteration] * jnp.square(q_values - batch_targets).mean(axis=0)
                )

        loss_mesh /= importance_iteration.sum()

        if self.add_infinity:
            fixed_point = self.fixed_point(pbo_params).reshape((1, -1))
            batch_targets = self.q.compute_target(fixed_point, sample).reshape(
                (-1, n_discrete_states, n_discrete_actions)
            )

            q_values = jax.vmap(lambda weights: self.q(self.q.to_params(weights), sample["state"], sample["action"]))(
                fixed_point
            ).reshape((-1, n_discrete_states, n_discrete_actions))

            if ord == 1:
                loss_mesh = loss_mesh.at[:].add(
                    importance_iteration[-1] * jnp.abs(q_values - batch_targets).mean(axis=0)
                )
            else:
                loss_mesh = loss_mesh.at[:].add(
                    importance_iteration[-1] * jnp.square(q_values - batch_targets).mean(axis=0)
                )

        return loss_mesh

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        raise NotImplementedError
