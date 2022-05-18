import haiku as hk
import jax.numpy as jnp
import jax

from pbo.networks.jax.q import BaseQ


class BasePBO:
    def __init__(self, network: hk.Module, network_key: int, gamma: float, q: BaseQ) -> None:
        self.gamma = gamma
        self.q = q

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, weights=jnp.zeros((q.weights_dimension)))

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

    def compute_target(self, batch: dict, weights: jnp.ndarray) -> jnp.ndarray:
        sample_batch_size = batch["reward"].shape[0]
        weights_batch_size = weights.shape[0]

        target = jnp.zeros((sample_batch_size * weights_batch_size, 1))

        for idx_weights in range(weights_batch_size):
            q_params = self.q.convert_to_params(weights[idx_weights])
            target_weights = batch["reward"] + self.gamma * self.q.max_value(q_params, batch["next_state"])

            target = target.at[idx_weights * sample_batch_size : (idx_weights + 1) * sample_batch_size].set(
                target_weights
            )

        return target

    def loss(self, pbo_params: hk.Params, batch: dict, weights: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        sample_batch_size = batch["reward"].shape[0]
        weights_batch_size = weights.shape[0]
        loss = 0

        iterated_weights = self.network.apply(pbo_params, weights)

        for idx_weights in range(weights_batch_size):
            iterated_q_params = self.q.convert_to_params(iterated_weights[idx_weights])

            loss += jnp.linalg.norm(
                self.q.network.apply(iterated_q_params, batch["state"], batch["action"])
                - target[idx_weights * sample_batch_size : (idx_weights + 1) * sample_batch_size],
                ord=1,
            )

        return loss

    def get_fixed_point(self) -> jnp.ndarray:
        raise NotImplementedError


class LinearPBONet(hk.Module):
    def __init__(self, layer_dimension: int) -> None:
        super().__init__(name="LinearNet")
        self.layer_dimension = layer_dimension

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.layer_dimension, name="linear")(weights)

        return x


class LinearPBO(BasePBO):
    def __init__(self, network_key: int, gamma: float, q: BaseQ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q.weights_dimension)(weights)

        super(LinearPBO, self).__init__(network, network_key, gamma, q)

    def get_fixed_point(self) -> jnp.ndarray:
        return (
            -jnp.linalg.inv(self.params["LinearNet/linear"]["w"] - jnp.eye(self.q.weights_dimension))
            @ self.params["LinearNet/linear"]["b"]
        )
