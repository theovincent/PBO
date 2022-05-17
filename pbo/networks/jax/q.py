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
        action_range_on_max: float,
        n_actions_on_max: int,
    ) -> None:
        self.random_weights_range = random_weights_range
        self.random_weights_key = random_weights_key
        self.action_range_on_max = action_range_on_max
        self.n_actions_on_max = n_actions_on_max

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, state=jnp.zeros((1)), action=jnp.zeros((1)))

        self.q_weights_dimensions = 0
        for layer in self.params.values():
            for weight in layer.values():
                self.q_weights_dimensions += np.prod(weight.shape)

        self.l1_loss_and_grad_loss = jax.jit(jax.value_and_grad(self.l1_loss))

    def get_random_weights(self) -> jnp.ndarray:
        self.random_weights_key, key = jax.random.split(self.random_weights_key)

        return jax.random.uniform(
            key, shape=(self.q_weights_dimensions,), minval=-self.random_weights_range, maxval=self.random_weights_range
        )

    def get_random_init_weights(self) -> jnp.ndarray:
        self.random_weights_key, key = jax.random.split(self.random_weights_key)

        return self.convert_to_weights(self.network.init(rng=key, state=jnp.zeros((1)), action=jnp.zeros((1))))

    def max_value(self, q_params: hk.Params, state: jnp.ndarray) -> jnp.ndarray:
        discrete_actions_on_max = jnp.linspace(
            -self.action_range_on_max, self.action_range_on_max, num=self.n_actions_on_max
        ).reshape((-1, 1))

        max_value_batch = np.zeros(state.shape[0])

        for idx_s, s in enumerate(state):
            max_value_batch[idx_s] = self.network.apply(
                q_params, s.repeat(self.n_actions_on_max).reshape((-1, 1)), discrete_actions_on_max
            ).max()

        return jnp.array(max_value_batch).reshape((-1, 1))

    def get_discrete_q(
        self, q_params: hk.Params, discrete_states: np.ndarray, discrete_actions: np.ndarray
    ) -> np.ndarray:
        q_values = np.zeros((len(discrete_states), len(discrete_actions)))

        for idx_state, state in enumerate(discrete_states):
            for idx_action, action in enumerate(discrete_actions):
                q_values[idx_state, idx_action] = self.network.apply(q_params, jnp.array([state]), jnp.array([action]))

        return q_values

    def convert_to_params(self, weight: jnp.ndarray) -> hk.Params:
        raise NotImplementedError

    def convert_to_weights(self, params: hk.Params) -> jnp.ndarray:
        raise NotImplementedError

    def l1_loss(self, q_params: hk.Params, state: jnp.ndarray, action: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(self.network.apply(q_params, state, action) - target).sum()


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
        action_range_on_max: float,
        n_actions_on_max: int,
    ) -> None:
        def network(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return FullyConnectedQNet(layer_dimension)(state, action)

        super(FullyConnectedQFunction, self).__init__(
            network, network_key, random_weights_range, random_weights_key, action_range_on_max, n_actions_on_max
        )

        self.weigths_information = {}
        current_idx = 0

        for key_layer, layer in self.params.items():
            self.weigths_information[key_layer] = dict()
            for key_weight_layer, weight_layer in layer.items():
                self.weigths_information[key_layer][key_weight_layer] = {
                    "begin_idx": current_idx,
                    "end_idx": current_idx + np.prod(weight_layer.shape),
                    "shape": weight_layer.shape,
                }
                current_idx += np.prod(weight_layer.shape)

    def convert_to_params(self, weights: jnp.ndarray) -> hk.Params:
        params = dict()

        for key_layer, layer in self.params.items():
            params[key_layer] = dict()
            for key_weight_layer in layer.keys():
                begin_idx = self.weigths_information[key_layer][key_weight_layer]["begin_idx"]
                end_idx = self.weigths_information[key_layer][key_weight_layer]["end_idx"]
                shape = self.weigths_information[key_layer][key_weight_layer]["shape"]

                params[key_layer][key_weight_layer] = weights[begin_idx:end_idx].reshape(shape)

        return params

    def convert_to_weights(self, params: hk.Params) -> jnp.ndarray:
        weights = []

        for layer in params.values():
            for weight_layer in layer.values():
                weights.extend(weight_layer.flatten())

        return jnp.array(weights)
