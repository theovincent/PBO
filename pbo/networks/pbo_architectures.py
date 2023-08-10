from typing import Sequence
import jax
import flax.linen as nn

from pbo.networks.base_q import BaseQ
from pbo.networks.base_pbo import BasePBO


class MLPNet(nn.Module):
    features: Sequence[int]
    weights_dimension: int

    @nn.compact
    def __call__(self, state):
        x = state
        for feature in self.features:
            x = nn.relu(nn.Dense(feature)(x))
        x = nn.Dense(self.weights_dimension)(x)

        return x


class MLPPBO(BasePBO):
    def __init__(
        self,
        q: BaseQ,
        max_bellman_iterations: int,
        features: Sequence[int],
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
        n_training_steps_per_current_weight_update: int,
    ) -> None:
        super().__init__(
            q,
            max_bellman_iterations,
            MLPNet(features, q.convert_params.weights_dimension),
            network_key,
            learning_rate,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_training_steps_per_current_weight_update,
        )
