from functools import partial

import numpy as np
import haiku as hk
import optax
import jax
import jax.numpy as jnp


class BaseQ:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actions_on_max: jnp.ndarray,
        gamma: float,
        network: hk.Module,
        network_key: int,
        learning_rate: dict,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actions_on_max = actions_on_max
        self.index_actions_on_max = jnp.arange(actions_on_max.shape[0])
        self.gamma = gamma
        self.network = hk.without_apply_rng(hk.transform(network))
        self.network_key = network_key
        self.params = self.network.init(
            rng=self.network_key, state=jnp.zeros((self.state_dim)), action=jnp.zeros((self.action_dim))
        )

        self.weights_information = {}
        self.weights_dimension = 0

        for key_layer, layer in self.params.items():
            self.weights_information[key_layer] = dict()
            for key_weight_layer, weight_layer in layer.items():
                # int because weight_layer.shape = () can happen
                weight_layer_dimensions = int(np.prod(weight_layer.shape))

                self.weights_information[key_layer][key_weight_layer] = {
                    "begin_idx": self.weights_dimension,
                    "end_idx": self.weights_dimension + weight_layer_dimensions,
                    "shape": weight_layer.shape,
                }
                self.weights_dimension += weight_layer_dimensions

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

        if learning_rate is not None:
            self.learning_rate_schedule = optax.linear_schedule(
                learning_rate["first"], learning_rate["last"], learning_rate["duration"]
            )
            self.optimizer = optax.adam(self.learning_rate_schedule)
            self.optimizer_state = self.optimizer.init(self.params)

    def random_init_weights(self) -> jnp.ndarray:
        self.network_key, key = jax.random.split(self.network_key)

        return self.to_weights(
            self.network.init(rng=key, state=jnp.zeros(self.state_dim), action=jnp.zeros(self.action_dim))
        )

    @partial(jax.jit, static_argnames="self")
    def to_params(self, weights: jnp.ndarray) -> hk.Params:
        params = dict()

        for key_layer, layer_info in self.weights_information.items():
            params[key_layer] = dict()
            for key_weight_layer, weight_layer_info in layer_info.items():
                begin_idx = weight_layer_info["begin_idx"]
                end_idx = weight_layer_info["end_idx"]
                shape = weight_layer_info["shape"]

                params[key_layer][key_weight_layer] = weights[begin_idx:end_idx].reshape(shape)

        return params

    @partial(jax.jit, static_argnames="self")
    def to_weights(self, params: hk.Params) -> jnp.ndarray:
        weights = jnp.zeros(self.weights_dimension)

        for key_layer, layer in params.items():
            for key_weight_layer, weight_layer in layer.items():
                begin_idx = self.weights_information[key_layer][key_weight_layer]["begin_idx"]
                end_idx = self.weights_information[key_layer][key_weight_layer]["end_idx"]

                weights = weights.at[begin_idx:end_idx].set(weight_layer.flatten())

        return jnp.array(weights)

    @partial(jax.jit, static_argnames="self")
    def __call__(self, params: hk.Params, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, states, actions)

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, weights: jnp.ndarray, samples: dict) -> jnp.ndarray:
        return jax.vmap(
            lambda weight: samples["reward"]
            + (1 - samples["absorbing"]) * self.gamma * self.max_value(self.to_params(weight), samples["next_state"])
        )(weights)

    @partial(jax.jit, static_argnames="self")
    def max_value(self, q_params: hk.Params, states: jnp.ndarray) -> jnp.ndarray:
        index_states = jnp.arange(states.shape[0])

        indexes_states_mesh, indexes_actions_mesh = jnp.meshgrid(index_states, self.index_actions_on_max, indexing="ij")
        states_ = states[indexes_states_mesh.flatten()]
        actions_ = self.actions_on_max[indexes_actions_mesh.flatten()]

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        max_values = self(q_params, states_, actions_).reshape((states.shape[0], self.actions_on_max.shape[0]))

        return max_values.max(axis=1).reshape((states.shape[0], 1))

    @partial(jax.jit, static_argnames=("self", "ord"))
    def loss(self, q_params: hk.Params, q_params_target, samples: dict, ord: int = 2) -> jnp.ndarray:
        target = self.compute_target(self.to_weights(q_params_target).reshape((1, self.weights_dimension)), samples)[0]

        if ord == 1:
            return jnp.abs(self(q_params, samples["state"], samples["action"]) - target).mean()
        else:
            return jnp.square(self(q_params, samples["state"], samples["action"]) - target).mean()

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: hk.Params, params_target: hk.Params, optimizer_state: tuple, batch_samples: jnp.ndarray
    ) -> tuple:
        loss, grad_loss = self.loss_and_grad(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def reset_optimizer(self) -> None:
        self.optimizer = optax.adam(self.learning_rate_schedule)
        self.optimizer_state = self.optimizer.init(self.params)


class BaseMultiHeadQ(BaseQ):
    def __init__(
        self,
        n_heads: int,
        state_dim: int,
        action_dim: int,
        actions_on_max: jnp.ndarray,
        gamma: float,
        network: hk.Module,
        network_key: int,
        learning_rate: dict,
    ) -> None:
        self.n_heads = n_heads
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            actions_on_max=actions_on_max,
            gamma=gamma,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, q_params: hk.Params, samples: dict) -> jnp.ndarray:
        return jnp.repeat(samples["reward"], self.n_heads, axis=1) + jnp.repeat(
            1 - samples["absorbing"], self.n_heads, axis=1
        ) * self.gamma * self.max_value(q_params, samples["next_state"])

    @partial(jax.jit, static_argnames="self")
    def max_value(self, q_params: hk.Params, states: jnp.ndarray) -> jnp.ndarray:
        index_states = jnp.arange(states.shape[0])

        indexes_states_mesh, indexes_actions_mesh = jnp.meshgrid(index_states, self.index_actions_on_max, indexing="ij")
        states_ = states[indexes_states_mesh.flatten()]
        actions_ = self.actions_on_max[indexes_actions_mesh.flatten()]

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        max_values = self(q_params, states_, actions_).reshape(
            (states.shape[0], self.actions_on_max.shape[0], self.n_heads)
        )

        return max_values.max(axis=1)

    @partial(jax.jit, static_argnames=("self", "ord"))
    def loss(self, q_params: hk.Params, q_params_target: hk.Params, samples: dict, ord: int = 2) -> jnp.ndarray:
        target = self.compute_target(q_params_target, samples)

        if ord == 1:
            return jnp.abs(self(q_params, samples["state"], samples["action"])[:, 1:] - target[:, :-1]).mean()
        else:
            return jnp.square(self(q_params, samples["state"], samples["action"])[:, 1:] - target[:, :-1]).mean()
