from typing import Sequence
import flax.linen as nn
import jax
import jax.numpy as jnp

from pbo.networks.base_q import BaseQ


class MLPNet(nn.Module):
    features: Sequence[int]
    n_actions: int

    @nn.compact
    def __call__(self, state):
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
        learning_rate: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
    ) -> None:
        super().__init__(
            {"state": jnp.zeros(state_shape, dtype=jnp.float32)},
            n_actions,
            gamma,
            MLPNet(features, n_actions),
            network_key,
            learning_rate,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
        )