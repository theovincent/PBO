from functools import partial

import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp


class BaseQ:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continuous_actions: bool,
        n_actions_on_max: int,
        action_range_on_max: float,
        gamma: float,
        network: hk.Module,
        network_key: int,
        random_weights_range: float,
        random_weights_key: int,
    ) -> None:
        self.gamma = gamma
        self.random_weights_range = random_weights_range
        self.random_weights_key = random_weights_key
        self.n_actions_on_max = n_actions_on_max
        self.index_actions_on_max = jnp.arange(n_actions_on_max)
        if continuous_actions:
            self.discrete_actions_on_max = jnp.linspace(
                -action_range_on_max, action_range_on_max, num=n_actions_on_max
            ).reshape((n_actions_on_max, action_dim))
        else:
            self.discrete_actions_on_max = jnp.arange(n_actions_on_max).reshape((n_actions_on_max, action_dim))

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, state=jnp.zeros((state_dim)), action=jnp.zeros((action_dim)))

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

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

    def random_weights(self) -> jnp.ndarray:
        self.random_weights_key, key = jax.random.split(self.random_weights_key)

        return jax.random.uniform(
            key, shape=(self.weights_dimension,), minval=-self.random_weights_range, maxval=self.random_weights_range
        )

    def random_init_weights(self) -> jnp.ndarray:
        self.random_weights_key, key = jax.random.split(self.random_weights_key)

        return self.to_weights(self.network.init(rng=key, state=jnp.zeros((1)), action=jnp.zeros((1))))

    @partial(jax.jit, static_argnames="self")
    def __call__(self, params: hk.Params, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, states, actions)

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, batch_weights: jnp.ndarray, batch_samples: dict) -> jnp.ndarray:
        return jax.vmap(
            lambda weights: batch_samples["reward"]
            + self.gamma * self.max_value(self.to_params(weights), batch_samples["next_state"])
        )(batch_weights)

    @partial(jax.jit, static_argnames="self")
    def max_value(self, q_params: hk.Params, batch_states: jnp.ndarray) -> jnp.ndarray:
        index_batch_states = jnp.arange(batch_states.shape[0])

        indexes_states_mesh, indexes_actions_mesh = jnp.meshgrid(
            index_batch_states, self.index_actions_on_max, indexing="ij"
        )
        states = batch_states[indexes_states_mesh.flatten()]
        actions = self.discrete_actions_on_max[indexes_actions_mesh.flatten()]

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        batch_values = self(q_params, states, actions).reshape((batch_states.shape[0], self.n_actions_on_max))

        return batch_values.max(axis=1).reshape((-1, 1))

    @partial(jax.jit, static_argnames="self")
    def discretize(
        self, batch_weights: jnp.ndarray, discrete_states: np.ndarray, discrete_actions: np.ndarray
    ) -> jnp.ndarray:
        states_mesh, actions_mesh = jnp.meshgrid(discrete_states, discrete_actions, indexing="ij")
        states = states_mesh.reshape((-1, 1))
        actions = actions_mesh.reshape((-1, 1))

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        return jax.vmap(lambda weights: self(self.to_params(weights), states, actions))(batch_weights).reshape(
            (-1, len(discrete_states), len(discrete_actions))
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

    @partial(jax.jit, static_argnames=("self", "ord"))
    def loss(self, q_params: hk.Params, q_params_target, sample: dict, ord: int = 2) -> jnp.ndarray:
        target = self.compute_target(self.to_weights(q_params_target).reshape((-1, self.weights_dimension)), sample)[0]

        if ord == 1:
            return jnp.abs(self(q_params, sample["state"], sample["action"]) - target).mean()
        else:
            return jnp.square(self(q_params, sample["state"], sample["action"]) - target).mean()
