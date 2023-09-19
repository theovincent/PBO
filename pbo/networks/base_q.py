from typing import Tuple, Union, Dict
from functools import partial
from flax.core import FrozenDict
import flax.linen as nn
import optax
import jax
import jax.numpy as jnp

from pbo.sample_collection.replay_buffer import ReplayBuffer
from pbo.utils.params import ParameterConverter, save_pickled_data


class BaseQ:
    def __init__(
        self,
        q_inputs: dict,
        n_actions: int,
        gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: Union[float, None],
        n_training_steps_per_online_update: Union[int, None],
        n_training_steps_per_target_update: Union[int, None],
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.network = network
        self.network_key = network_key
        self.q_inputs = q_inputs
        self.params = self.network.init(self.network_key, **self.q_inputs)
        self.convert_params = ParameterConverter(self.params)

        if learning_rate is not None:
            self.target_params = self.params

            self.n_training_steps_per_online_update = n_training_steps_per_online_update
            self.n_training_steps_per_target_update = n_training_steps_per_target_update

            self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

            self.optimizer = optax.adam(learning_rate)
            self.optimizer_state = self.optimizer.init(self.params)

    @partial(jax.jit, static_argnames="self")
    def apply(self, params: FrozenDict, states: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, states)

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: FrozenDict, samples: Dict) -> jnp.ndarray:
        return samples["reward"] + (1 - samples["absorbing"]) * self.gamma * self.apply(
            params, samples["next_state"]
        ).max(axis=1)

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: Dict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)
        q_states_actions = self.apply(params, samples["state"])

        # mapping over the states
        predictions = jax.vmap(lambda q_state_actions, action: q_state_actions[action])(
            q_states_actions, samples["action"]
        )

        return self.metric(predictions - targets, ord="2")

    @staticmethod
    def metric(error: jnp.ndarray, ord: str) -> jnp.float32:
        if ord == "huber":
            return optax.huber_loss(error, 0).mean()
        elif ord == "1":
            return jnp.abs(error).mean()
        elif ord == "2":
            return jnp.square(error).mean()
        elif ord == "sum":
            return jnp.square(error).sum()

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: FrozenDict, params_target: FrozenDict, optimizer_state: Tuple, batch_samples: Dict
    ) -> Tuple[FrozenDict, FrozenDict, jnp.float32]:
        loss, grad_loss = self.loss_and_grad(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        if step % self.n_training_steps_per_online_update == 0:
            batch_samples = replay_buffer.sample_random_batch(key)

            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )

            return loss
        else:
            return jnp.nan

    def update_target_params(self, step: int) -> None:
        if step % self.n_training_steps_per_target_update == 0:
            self.target_params = self.params

    @partial(jax.jit, static_argnames="self")
    def random_action(self, key: jax.random.PRNGKeyArray) -> jnp.int8:
        return jax.random.choice(key, jnp.arange(self.n_actions)).astype(jnp.int8)

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.int8:
        # key is not used here
        return jnp.argmax(self.apply(params, jnp.array(state, dtype=jnp.float32))).astype(jnp.int8)

    def draw_current_batch_weights(self, n_weights: int) -> jnp.ndarray:
        weights = jnp.zeros((n_weights, self.convert_params.weights_dimension))
        ### The first set of weights is the one FQI would have i.e. generate with the same seed.
        weights = weights.at[0].set(self.convert_params.to_weights(self.params))

        for i in range(1, n_weights):
            self.network_key, key = jax.random.split(self.network_key)
            weights = weights.at[i].set(self.convert_params.to_weights(self.network.init(key, **self.q_inputs)))

        return weights

    def save(self, path: str) -> None:
        save_pickled_data(path + "_online_params", self.params)
