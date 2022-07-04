from functools import partial

import haiku as hk
import optax
import jax
import jax.numpy as jnp

from pbo.networks.base_q import BaseQ


class LearnableQ(BaseQ):
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
        learning_rate: dict,
    ) -> None:
        super().__init__(
            state_dim,
            action_dim,
            continuous_actions,
            n_actions_on_max,
            action_range_on_max,
            gamma,
            network,
            network_key,
            random_weights_range,
            random_weights_key,
        )

        if learning_rate is not None:
            self.learning_rate_schedule = optax.linear_schedule(
                learning_rate["first"], learning_rate["last"], learning_rate["duration"]
            )
            self.optimizer = optax.adam(self.learning_rate_schedule)
            self.optimizer_state = self.optimizer.init(self.params)

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


class FullyConnectedQNet(hk.Module):
    def __init__(self, layers_dimension: list, zero_initializer: bool = False) -> None:
        super().__init__(name="FullyConnectedNet")
        self.layers_dimension = layers_dimension
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
            x = hk.Linear(layer_dimension, name=f"linear_{idx}")(x)
            x = jax.nn.tanh(x)

        x = hk.Linear(1, w_init=self.initializer, name="linear_last")(x)

        return x


class FullyConnectedQ(LearnableQ):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continuous_actions: bool,
        n_actions_on_max: int,
        action_range_on_max: float,
        gamma: float,
        network_key: int,
        random_weights_range: float,
        random_weights_key: int,
        learning_rate: dict,
        layers_dimension: list,
        zero_initializer: bool,
    ) -> None:
        def network(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return FullyConnectedQNet(layers_dimension, zero_initializer)(state, action)

        super().__init__(
            state_dim,
            action_dim,
            continuous_actions,
            n_actions_on_max,
            action_range_on_max,
            gamma,
            network,
            network_key,
            random_weights_range,
            random_weights_key,
            learning_rate,
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


class LQRQ(LearnableQ):
    def __init__(
        self,
        gamma: float,
        network_key: int,
        random_weights_range: float,
        random_weights_key: int,
        action_range_on_max: float,
        n_actions_on_max: int,
        learning_rate: dict = None,
    ) -> None:
        def network(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return LQRQNet()(state, action)

        super().__init__(
            gamma,
            network,
            network_key,
            random_weights_range,
            random_weights_key,
            action_range_on_max,
            n_actions_on_max,
            learning_rate,
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
            self.initializer = jnp.zeros
        else:
            self.initializer = hk.initializers.TruncatedNormal()

    def __call__(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        table = hk.get_parameter("table", (self.n_states, self.n_actions), state.dtype, init=self.initializer)

        return jax.vmap(lambda state_, action_: table[state_, action_])(state.astype(int), action.astype(int))


class TableQ(LearnableQ):
    def __init__(
        self,
        state_dim: int,
        n_states: int,
        action_dim: int,
        n_actions: int,
        gamma: float,
        network_key: int,
        random_weights_range: float,
        random_weights_key: int,
        learning_rate: dict,
        zero_initializer: bool,
    ) -> None:
        def network(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return TableQNet(n_states, n_actions, zero_initializer)(state, action)

        super().__init__(
            state_dim,
            action_dim,
            False,
            n_actions,
            None,
            gamma,
            network,
            network_key,
            random_weights_range,
            random_weights_key,
            learning_rate,
        )
