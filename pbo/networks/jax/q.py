import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp


class BaseQ:
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
        self.n_actions_on_max = n_actions_on_max
        self.discrete_actions_on_max = jnp.linspace(-action_range_on_max, action_range_on_max, num=n_actions_on_max)

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, state=jnp.zeros((1)), action=jnp.zeros((1)))

        self.weights_dimension = 0
        for layer in self.params.values():
            for weight in layer.values():
                self.weights_dimension += np.prod(weight.shape)

        self.l1_loss_and_grad = jax.jit(jax.value_and_grad(self.l1_loss))

    def random_weights(self) -> jnp.ndarray:
        self.random_weights_key, key = jax.random.split(self.random_weights_key)

        return jax.random.uniform(
            key, shape=(self.weights_dimension,), minval=-self.random_weights_range, maxval=self.random_weights_range
        )

    def random_init_weights(self) -> jnp.ndarray:
        self.random_weights_key, key = jax.random.split(self.random_weights_key)

        return self.to_weights(self.network.init(rng=key, state=jnp.zeros((1)), action=jnp.zeros((1))))

    def max_value(self, q_params: hk.Params, batch_states: jnp.ndarray) -> jnp.ndarray:
        states_mesh, actions_mesh = jnp.meshgrid(batch_states.flatten(), self.discrete_actions_on_max, indexing="ij")
        states = states_mesh.reshape((-1, 1))
        actions = actions_mesh.reshape((-1, 1))

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        batch_values = self.network.apply(q_params, states, actions).reshape(
            (batch_states.shape[0], self.n_actions_on_max)
        )

        return batch_values.max(axis=1).reshape((-1, 1))

    def discretize(
        self, batch_weights: jnp.ndarray, discrete_states: np.ndarray, discrete_actions: np.ndarray
    ) -> jnp.ndarray:
        states_mesh, actions_mesh = jnp.meshgrid(discrete_states, discrete_actions, indexing="ij")
        states = states_mesh.reshape((-1, 1))
        actions = actions_mesh.reshape((-1, 1))

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        return jax.vmap(lambda weights: self.network.apply(self.to_params(weights), states, actions))(
            batch_weights
        ).reshape((-1, len(discrete_states), len(discrete_actions)))

    def to_params(self, weights: jnp.ndarray) -> hk.Params:
        raise NotImplementedError

    def to_weights(self, params: hk.Params) -> jnp.ndarray:
        raise NotImplementedError

    def l1_loss(self, q_params: hk.Params, state: jnp.ndarray, action: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(self.network.apply(q_params, state, action) - target, ord=1)


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


class FullyConnectedQ(BaseQ):
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

        super(FullyConnectedQ, self).__init__(
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

    def to_params(self, weights: jnp.ndarray) -> hk.Params:
        params = dict()

        for key_layer, layer in self.params.items():
            params[key_layer] = dict()
            for key_weight_layer in layer.keys():
                begin_idx = self.weigths_information[key_layer][key_weight_layer]["begin_idx"]
                end_idx = self.weigths_information[key_layer][key_weight_layer]["end_idx"]
                shape = self.weigths_information[key_layer][key_weight_layer]["shape"]

                params[key_layer][key_weight_layer] = weights[begin_idx:end_idx].reshape(shape)

        return params

    def to_weights(self, params: hk.Params) -> jnp.ndarray:
        weights = []

        for layer in params.values():
            for weight_layer in layer.values():
                weights.extend(weight_layer.flatten())

        return jnp.array(weights)
