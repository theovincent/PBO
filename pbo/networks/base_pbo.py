from typing import Tuple, Union, Dict
from functools import partial
from flax.core import FrozenDict
import flax.linen as nn
import optax
import jax.numpy as jnp
import jax

from pbo.networks.base_q import BaseQ
from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.utils.params import save_pickled_data


class BasePBO:
    def __init__(
        self,
        q: BaseQ,
        bellman_iterations_scope: int,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: Union[float, None],
        n_training_steps_per_online_update: Union[int, None],
        n_training_steps_per_target_update: Union[int, None],
        n_current_weights: Union[int, None],
        n_training_steps_per_current_weight_update: Union[int, None],
    ) -> None:
        self.q = q
        self.bellman_iterations_scope = bellman_iterations_scope
        self.network = network
        self.network_key = network_key
        self.params = self.network.init(
            self.network_key, weights=jnp.zeros(self.q.convert_params.weights_dimension, dtype=jnp.float32)
        )

        if learning_rate is not None:
            self.target_params = self.params

            self.n_training_steps_per_online_update = n_training_steps_per_online_update
            self.n_training_steps_per_target_update = n_training_steps_per_target_update
            self.n_current_weights = n_current_weights
            self.current_batch_weights = self.q.draw_current_batch_weights(self.n_current_weights)
            self.n_training_steps_per_current_weight_update = n_training_steps_per_current_weight_update

            self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

            self.optimizer = optax.adam(learning_rate)
            self.optimizer_state = self.optimizer.init(self.params)
        else:
            # We define the current weights for being able to sample actions.
            self.n_current_weights = 1
            self.current_batch_weights = self.q.draw_current_batch_weights(self.n_current_weights)

    @partial(jax.jit, static_argnames="self")
    def apply(self, params: FrozenDict, weights: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, weights)

    @partial(jax.jit, static_argnames="self")
    def td_error(self, weights: jnp.ndarray, weights_target: jnp.ndarray, samples: Dict) -> float:
        # Compute the td error for each pair of weight and target weight and then take the mean over the computed values
        return jnp.mean(
            jax.vmap(
                lambda weights_, weights_target_: self.q.loss(
                    self.q.convert_params.to_params(weights_),
                    self.q.convert_params.to_params(weights_target_),
                    samples,
                )
            )(weights, weights_target)
        )

    @partial(jax.jit, static_argnames="self")
    def loss(self, pbo_params: FrozenDict, pbo_params_target: FrozenDict, weights: jnp.ndarray, samples: Dict) -> float:
        iterated_weights_target = weights
        iterated_weights = self.apply(pbo_params, weights)

        loss = self.td_error(iterated_weights, iterated_weights_target, samples)

        for _ in jnp.arange(1, self.bellman_iterations_scope):
            iterated_weights_target = self.apply(pbo_params_target, iterated_weights_target)

            # Uncomment to limit the back propagation to one iteration
            # iterated_weights = jax.lax.stop_gradient(iterated_weights)
            iterated_weights = self.apply(pbo_params, iterated_weights)

            loss += self.td_error(iterated_weights, iterated_weights_target, samples)

        return loss

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        optimizer_state: tuple,
        batch_weights: jnp.ndarray,
        batch_samples: jnp.ndarray,
    ) -> Tuple[FrozenDict, FrozenDict, jnp.float32]:
        loss, grad_loss = self.loss_and_grad(params, params_target, batch_weights, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def update_current_weights(self, params: FrozenDict) -> None:
        self.current_batch_weights = self.apply(params, self.current_batch_weights)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        if step % self.n_training_steps_per_current_weight_update == 0:
            self.update_current_weights(self.params)

        if step % self.n_training_steps_per_online_update == 0:
            batch_samples = replay_buffer.sample_random_batch(key)

            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, self.current_batch_weights, batch_samples
            )

            return loss
        else:
            return jnp.nan

    def update_target_params(self, step: int) -> None:
        if (step % self.n_training_steps_per_target_update == 0) or (
            step % self.n_training_steps_per_current_weight_update == 0
        ):
            self.target_params = self.params

    @partial(jax.jit, static_argnames="self")
    def random_action(self, key: jax.random.PRNGKeyArray) -> jnp.int8:
        return self.q.random_action(key)

    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.int8:
        # Decide the number of bellman iteration.
        key, iteration_key = jax.random.split(key)
        n_iterations = jax.random.choice(iteration_key, jnp.arange(self.bellman_iterations_scope))

        iterated_weigths = self.current_batch_weights[0]
        for _ in jnp.arange(n_iterations):
            iterated_weigths = self.apply(params, iterated_weigths)

        return self.q.best_action(self.q.convert_params.to_params(iterated_weigths), state, key)

    def save(self, path: str) -> None:
        save_pickled_data(path + "_online_params", self.params)
        save_pickled_data(path + "_current_batch_weights", self.current_batch_weights)
