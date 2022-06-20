from functools import partial

import haiku as hk
import jax.numpy as jnp
import jax

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
