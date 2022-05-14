from click import password_option
import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp


class BaseQFunction:
    def __init__(
        self,
        network: hk.Module,
        network_key: int,
        random_weights_range: float,
        random_weights_key: int,
        max_action: float,
        n_discrete_actions: int,
    ) -> None:
        self.random_weights_range = random_weights_range
        self.random_weights_key = random_weights_key
        self.max_action = max_action
        self.n_discrete_actions = n_discrete_actions

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, state=jnp.zeros((1)), action=jnp.zeros((1)))

        self.q_weights_dimensions = 0
        for layer in self.params.values():
            for weight in layer.values():
                self.q_weights_dimensions += np.prod(weight.shape)

    def get_random_weights(self) -> jnp.ndarray:
        self.random_weights_key, rng = jax.random.split(self.random_weights_key)

        return jax.random.uniform(
            rng, shape=(self.q_weights_dimensions,), minval=-self.random_weights_range, maxval=self.random_weights_range
        )

    def max_value(self, q_params: hk.Params, state: jnp.ndarray) -> jnp.ndarray:
        discrete_actions = jnp.linspace(-self.max_action, self.max_action, num=self.n_discrete_actions).reshape((-1, 1))

        max_value_batch = jnp.zeros(state.shape[0])

        for idx_s, s in enumerate(state):
            max_value_batch[idx_s] = self.network.apply(
                q_params, s.repeat(self.n_discrete_actions).reshape((-1, 1)), discrete_actions
            ).max()

        return max_value_batch.reshape((-1, 1))

    def get_discrete_q(self, q_params: hk.Params, max_discrete_state: float, n_discrete_states: int) -> np.ndarray:
        discrete_states = np.linspace(-max_discrete_state, max_discrete_state, n_discrete_states)
        discrete_actions = np.linspace(-self.max_action, self.max_action, self.n_discrete_actions)

        q_values = np.zeros((len(discrete_states), len(discrete_actions)))

        for idx_state, state in enumerate(discrete_states):
            for idx_action, action in enumerate(discrete_actions):
                q_values[idx_state, idx_action] = self.network.apply(q_params, jnp.array([state]), jnp.array([action]))

        return q_values

    def convert_to_params(self, weight: jnp.ndarray) -> hk.Params:
        params = dict()
        current_idx = 0

        for key_layer, layer in self.params.items():
            params[key_layer] = dict()
            for key_weight_layer, weight_layer in layer.items():
                params[key_layer][key_weight_layer] = weight[
                    current_idx : current_idx + jnp.prod(weight_layer.shape)
                ].reshape(weight_layer.shape)

                current_idx += jnp.prod(weight_layer.shape)

        return params


class FullyConnectedQNet(hk.Module):
    def __init__(self, layer_dimension: int) -> None:
        super().__init__(name="FullyConnectedNet")
        self.layer_dimension = layer_dimension

    def __call__(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        stacked_state_action = jnp.hstack((state, action))

        x = hk.Linear(self.layer_dimension, name="linear_1")(stacked_state_action)
        x = jax.nn.relu(x)
        x = hk.Linear(1, name="linear_2")(x)

        return x


class FullyConnectedQFunction(BaseQFunction):
    def __init__(
        self,
        layer_dimension: int,
        network_key: int,
        random_weights_range: float,
        random_weights_key: int,
        max_action: float,
        n_discrete_actions: int,
    ) -> None:
        def network(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return FullyConnectedQNet(layer_dimension)(state, action)

        super(FullyConnectedQFunction, self).__init__(
            network, network_key, random_weights_range, random_weights_key, max_action, n_discrete_actions
        )
