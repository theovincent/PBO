from typing import Sequence, Union, Dict
import jax
import jax.numpy as jnp
import flax.linen as nn

from pbo.networks.base_q import BaseQ
from pbo.networks.base_pbo import BasePBO


class MLPNet(nn.Module):
    features: Sequence
    weights_dimension: int
    initial_std: float

    @nn.compact
    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        initializer = (
            nn.initializers.variance_scaling(self.initial_std, "fan_in", "truncated_normal")
            if self.initial_std is not None
            else nn.linear.default_kernel_init
        )

        x = weights
        for feature in self.features:
            x = nn.relu(
                nn.Dense(
                    int(feature * self.weights_dimension),
                    kernel_init=initializer,
                )(x)
            )
        x = nn.Dense(self.weights_dimension, kernel_init=initializer)(x)

        return x


class MLPPBO(BasePBO):
    def __init__(
        self,
        q: BaseQ,
        bellman_iterations_scope: int,
        features: Sequence[int],
        network_key: jax.random.PRNGKeyArray,
        learning_rate: Union[Dict, float, None] = None,
        epsilon_optimizer: Union[float, None] = None,
        n_training_steps_per_online_update: Union[int, None] = None,
        n_training_steps_per_target_update: Union[int, None] = None,
        n_current_weights: Union[int, None] = None,
        n_training_steps_per_current_weights_update: Union[int, None] = None,
        initial_std: Union[float, None] = None,
    ) -> None:
        super().__init__(
            q,
            bellman_iterations_scope,
            MLPNet(features, q.convert_params.weights_dimension, initial_std),
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_current_weights,
            n_training_steps_per_current_weights_update,
        )


class SplittedMLPNet(nn.Module):
    split_size: int
    features: Sequence[int]
    weights_dimension: int

    @nn.compact
    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        batch_weights = jnp.atleast_2d(weights)
        applied_weights = jnp.zeros_like(batch_weights)

        for idx_split in range(0, self.weights_dimension, self.split_size):
            applied_weights = applied_weights.at[:, idx_split : idx_split + self.split_size].set(
                self.apply_on_splitted_weigths(batch_weights[:, idx_split : idx_split + self.split_size])
            )

        return applied_weights.reshape(weights.shape)

    def apply_on_splitted_weigths(self, splitted_weights):
        splitted_weights_dimension = splitted_weights.shape[1]
        x = splitted_weights

        for feature in self.features:
            x = nn.relu(nn.Dense(int(feature * splitted_weights_dimension))(x))
        x = nn.Dense(splitted_weights_dimension)(x)

        return x


class SplittedMLPPBO(BasePBO):
    def __init__(
        self,
        q: BaseQ,
        bellman_iterations_scope: int,
        split_size: int,
        features: Sequence[int],
        network_key: jax.random.PRNGKeyArray,
        learning_rate: Union[Dict, float, None] = None,
        epsilon_optimizer: Union[float, None] = None,
        n_training_steps_per_online_update: Union[int, None] = None,
        n_training_steps_per_target_update: Union[int, None] = None,
        n_current_weights: Union[int, None] = None,
        n_training_steps_per_current_weights_update: Union[int, None] = None,
    ) -> None:
        super().__init__(
            q,
            bellman_iterations_scope,
            SplittedMLPNet(split_size, features, q.convert_params.weights_dimension),
            network_key,
            learning_rate,
            epsilon_optimizer,
            n_training_steps_per_online_update,
            n_training_steps_per_target_update,
            n_current_weights,
            n_training_steps_per_current_weights_update,
        )
