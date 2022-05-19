import haiku as hk
import jax.numpy as jnp
import jax
import optax

from pbo.networks.jax.q import BaseQ


class BasePBO:
    def __init__(self, network: hk.Module, network_key: int, gamma: float, q: BaseQ, learning_rate: float) -> None:
        self.gamma = gamma
        self.q = q

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, weights=jnp.zeros((q.weights_dimension)))

        self.compute_target = jax.jit(self.compute_target_)
        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

        self.optimizer = optax.sgd(learning_rate=learning_rate)
        self.optimizer_state = self.optimizer.init(self.params)

    def compute_target_(self, batch_samples: dict, batch_weights: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda weights: batch_samples["reward"]
            + self.gamma * self.q.max_value(self.q.to_params(weights), batch_samples["next_state"])
        )(batch_weights)

    def loss(
        self, pbo_params: hk.Params, sample: dict, batch_weights: jnp.ndarray, batch_targets: jnp.ndarray
    ) -> jnp.ndarray:
        batch_iterated_weights = self.network.apply(pbo_params, batch_weights)

        q_values = jax.vmap(
            lambda weights: self.q.network.apply(self.q.to_params(weights), sample["state"], sample["action"])
        )(batch_iterated_weights)

        return jnp.linalg.norm((q_values - batch_targets).flatten(), ord=1)

    def learn_on_batch(self, batch_samples: jnp.ndarray, batch_weights: jnp.ndarray) -> jnp.ndarray:
        batch_targets = self.compute_target(batch_samples, batch_weights)

        loss, grad_loss = self.loss_and_grad(self.params, batch_samples, batch_weights, batch_targets)
        updates, self.optimizer_state = self.optimizer.update(grad_loss, self.optimizer_state)
        self.params = optax.apply_updates(self.params, updates)

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
    def __init__(self, network_key: int, gamma: float, q: BaseQ, learning_rate: float) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q.weights_dimension)(weights)

        super(LinearPBO, self).__init__(network, network_key, gamma, q, learning_rate)

    def get_fixed_point(self) -> jnp.ndarray:
        return (
            -jnp.linalg.inv(self.params["LinearNet/linear"]["w"] - jnp.eye(self.q.weights_dimension))
            @ self.params["LinearNet/linear"]["b"]
        )
