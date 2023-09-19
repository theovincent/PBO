from typing import Sequence, Union
import jax
import jax.numpy as jnp
import flax.linen as nn

from pbo.networks.base_q import BaseQ
from pbo.networks.base_pbo import BasePBO


class MLPNet(nn.Module):
    features: Sequence[int]
    weights_dimension: int

    @nn.compact
    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = weights
        for feature in self.features:
            x = nn.relu(nn.Dense(int(feature * self.weights_dimension))(x))
        x = nn.Dense(self.weights_dimension)(x)

        return x


class MLPPBO(BasePBO):
    def __init__(
        self,
        q: BaseQ,
        bellman_iterations_scope: int,
        features: Sequence[int],
        network_key: jax.random.PRNGKeyArray,
        learning_rate: Union[float, None] = None,
        n_training_steps_per_online_update: Union[int, None] = None,
        n_training_steps_per_target_update: Union[int, None] = None,
        n_current_weights: Union[int, None] = None,
        n_training_steps_per_current_weight_update: Union[int, None] = None,
    ) -> None:
        super().__init__(
            q,
            bellman_iterations_scope,
            MLPNet(features, q.convert_params.weights_dimension),
            network_key,
            learning_rate,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_current_weights,
            n_training_steps_per_current_weight_update,
        )
