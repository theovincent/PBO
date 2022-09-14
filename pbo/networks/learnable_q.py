import haiku as hk
import jax
import jax.numpy as jnp

from pbo.networks.base_q import BaseQ


class FullyConnectedQNet(hk.Module):
    def __init__(self, layers_dimension: list, initial_weight_std: float, zero_initializer: bool = False) -> None:
        super().__init__(name="FullyConnectedNet")
        self.layers_dimension = layers_dimension
        self.initial_weight_std = initial_weight_std
        if zero_initializer:
            self.initializer = hk.initializers.Constant(0)
        else:
            self.initializer = hk.initializers.TruncatedNormal()

    def __call__(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        x = jnp.hstack((state, action))

        for idx, layer_dimension in enumerate(self.layers_dimension, start=1):
            x = hk.Linear(
                layer_dimension,
                w_init=hk.initializers.TruncatedNormal(stddev=self.initial_weight_std),
                name=f"linear_{idx}",
            )(x)
            x = jax.nn.relu(x)

        x = hk.Linear(1, w_init=self.initializer, name="linear_last")(x)

        return x


class FullyConnectedQ(BaseQ):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actions_on_max: jnp.ndarray,
        gamma: float,
        network_key: int,
        layers_dimension: list,
        initial_weight_std: float,
        zero_initializer: bool,
        learning_rate: dict = None,
    ) -> None:
        def network(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return FullyConnectedQNet(layers_dimension, initial_weight_std, zero_initializer)(state, action)

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            actions_on_max=actions_on_max,
            gamma=gamma,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )


class LQRQNet(hk.Module):
    def __init__(self, zero_initializer: bool) -> None:
        super().__init__(name="LQRQNet")

        if zero_initializer:
            self.initializer = hk.initializers.Constant(0)
        else:
            self.initializer = hk.initializers.TruncatedNormal()

    def __call__(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        k = hk.get_parameter("k", (), state.dtype, init=self.initializer)
        i = hk.get_parameter("i", (), state.dtype, init=self.initializer)
        m = hk.get_parameter("m", (), state.dtype, init=self.initializer)

        return state**2 * k + 2 * state * action * i + action**2 * m


class LQRQ(BaseQ):
    def __init__(
        self,
        n_actions_on_max: int,
        max_action_on_max: float,
        network_key: int,
        zero_initializer: bool,
        learning_rate: dict = None,
    ) -> None:
        def network(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return LQRQNet(zero_initializer)(state, action)

        super().__init__(
            state_dim=1,
            action_dim=1,
            actions_on_max=jnp.linspace(-max_action_on_max, max_action_on_max, n_actions_on_max).reshape(
                (n_actions_on_max, 1)
            ),
            gamma=1,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )


class TableQNet(hk.Module):
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        zero_initializer: bool = False,
    ) -> None:
        super().__init__(name="TableQNet")
        self.n_states = n_states
        self.n_actions = n_actions
        if zero_initializer:
            self.initializer = hk.initializers.Constant(0)
        else:
            self.initializer = hk.initializers.TruncatedNormal()

    def __call__(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        table = hk.get_parameter("table", (self.n_states, self.n_actions), state.dtype, init=self.initializer)

        return jax.vmap(lambda state_, action_: table[state_, action_])(state.astype(int), action.astype(int))


class TableQ(BaseQ):
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float,
        network_key: int,
        zero_initializer: bool,
        learning_rate: dict = None,
    ) -> None:
        def network(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return TableQNet(n_states, n_actions, zero_initializer)(state, action)

        super().__init__(
            state_dim=1,
            action_dim=1,
            actions_on_max=jnp.arange(n_actions).reshape((n_actions, 1)),
            gamma=gamma,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )
