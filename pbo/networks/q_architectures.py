from typing import Sequence, Union
import flax.linen as nn
import jax
import jax.numpy as jnp

from pbo.networks.base_q import BaseQ


class MLPNet(nn.Module):
    features: Sequence[int]
    n_actions: int

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = state
        for feature in self.features:
            x = nn.relu(nn.Dense(feature)(x))
        x = nn.Dense(self.n_actions)(x)

        return x


class MLPQ(BaseQ):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        features: Sequence[int],
        network_key: jax.random.PRNGKeyArray,
        learning_rate: Union[float, None] = None,
        epsilon_optimizer: Union[float, None] = None,
        n_training_steps_per_online_update: Union[int, None] = None,
        n_training_steps_per_target_update: Union[int, None] = None,
    ) -> None:
        super().__init__(
            {"state": jnp.zeros(state_shape, dtype=jnp.float32)},
            n_actions,
            gamma,
            MLPNet(features, n_actions),
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )
