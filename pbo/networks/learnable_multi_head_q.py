import haiku as hk
import jax
import jax.numpy as jnp

from pbo.networks.base_q import BaseMultiHeadQ


class FullyConnectedMultiHeadQNet(hk.Module):
    def __init__(self, n_heads: int, layers_dimension: list, zero_initializer: bool = False) -> None:
        super().__init__(name="FullyConnectedNet")
        self.n_heads = n_heads
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
        x = jnp.atleast_2d(x)
        output = jnp.zeros((x.shape[0], self.n_heads))

        for idx_head in range(self.n_heads):
            for idx, layer_dimension in enumerate(self.layers_dimension, start=1):
                x = hk.Linear(layer_dimension, name=f"head_{idx_head}_linear_{idx}")(x)
                x = jax.nn.relu(x)

            output = output.at[:, idx_head].set(
                hk.Linear(1, w_init=self.initializer, name=f"head_{idx_head}_linear_last")(x)[:, 0]
            )

        return output


class FullyConnectedMultiHeadQ(BaseMultiHeadQ):
    def __init__(
        self,
        n_heads: int,
        state_dim: int,
        action_dim: int,
        actions_on_max: jnp.ndarray,
        gamma: float,
        network_key: int,
        layers_dimension: list,
        zero_initializer: bool,
        learning_rate: dict = None,
    ) -> None:
        def network(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return FullyConnectedMultiHeadQNet(n_heads, layers_dimension, zero_initializer)(state, action)

        super().__init__(
            n_heads,
            state_dim=state_dim,
            action_dim=action_dim,
            actions_on_max=actions_on_max,
            gamma=gamma,
            network=network,
            network_key=network_key,
            learning_rate=learning_rate,
        )
