from functools import partial

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
        continuous_actions: bool = True,
    ) -> None:
        self.random_weights_range = random_weights_range
        self.random_weights_key = random_weights_key
        self.n_actions_on_max = n_actions_on_max
        if continuous_actions:
            self.discrete_actions_on_max = jnp.linspace(-action_range_on_max, action_range_on_max, num=n_actions_on_max)
        else:
            self.discrete_actions_on_max = jnp.arange(n_actions_on_max)

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, state=jnp.zeros((1)), action=jnp.zeros((1)))

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
    def __call__(self, params: hk.Params, weights: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, weights)

    @partial(jax.jit, static_argnames="self")
    def max_value(self, q_params: hk.Params, batch_states: jnp.ndarray) -> jnp.ndarray:
        states_mesh, actions_mesh = jnp.meshgrid(batch_states.flatten(), self.discrete_actions_on_max, indexing="ij")
        states = states_mesh.reshape((-1, 1))
        actions = actions_mesh.reshape((-1, 1))

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

    @partial(jax.jit, static_argnames="self")
    def loss(self, q_params: hk.Params, state: jnp.ndarray, action: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(self(q_params, state, action) - target)


class FullyConnectedQNet(hk.Module):
    def __init__(self, layers_dimension: list) -> None:
        super().__init__(name="FullyConnectedNet")
        self.layers_dimension = layers_dimension

    def __call__(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        x = jnp.hstack((state, action))

        for idx, layer_dimension in enumerate(self.layers_dimension, start=1):
            x = hk.Linear(layer_dimension, name=f"linear_{idx}")(x)
            x = jax.nn.relu(x)

        x = hk.Linear(1, name="linear_last")(x)

        return x


class FullyConnectedQ(BaseQ):
    def __init__(
        self,
        layers_dimension: list,
        network_key: int,
        random_weights_range: float,
        random_weights_key: int,
        action_range_on_max: float,
        n_actions_on_max: int,
    ) -> None:
        def network(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return FullyConnectedQNet(layers_dimension)(state, action)

        super().__init__(
            network, network_key, random_weights_range, random_weights_key, action_range_on_max, n_actions_on_max
        )


class LQRQNet(hk.Module):
    def __init__(self) -> None:
        super().__init__(name="Theoretical3DQNet")

    def __call__(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        k = hk.get_parameter("k", (), state.dtype, init=hk.initializers.TruncatedNormal())
        i = hk.get_parameter("i", (), state.dtype, init=hk.initializers.TruncatedNormal())
        m = hk.get_parameter("m", (), state.dtype, init=hk.initializers.TruncatedNormal())

        return state**2 * k + 2 * state * action * i + action**2 * m


class LQRQ(BaseQ):
    def __init__(
        self,
        network_key: int,
        random_weights_range: float,
        random_weights_key: int,
        action_range_on_max: float,
        n_actions_on_max: int,
    ) -> None:
        def network(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return LQRQNet()(state, action)

        super().__init__(
            network, network_key, random_weights_range, random_weights_key, action_range_on_max, n_actions_on_max
        )


class TableQNet(hk.Module):
    def __init__(self, n_states: int, n_actions: int) -> None:
        super().__init__(name="TableQNet")
        self.n_states = n_states
        self.n_actions = n_actions

    def __call__(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        table = hk.get_parameter(
            "table", (self.n_states, self.n_actions), state.dtype, init=hk.initializers.TruncatedNormal()
        )

        return jax.vmap(lambda state_, action_: table[state_, action_])(state.astype(int), action.astype(int))


class TableQ(BaseQ):
    def __init__(
        self,
        network_key: int,
        random_weights_range: float,
        random_weights_key: int,
        n_states: int,
        n_actions: int,
    ) -> None:
        def network(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return TableQNet(n_states, n_actions)(state, action)

        super().__init__(
            network,
            network_key,
            random_weights_range,
            random_weights_key,
            None,
            n_actions,
            continuous_actions=False,
        )
