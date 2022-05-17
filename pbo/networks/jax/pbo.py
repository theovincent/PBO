import haiku as hk
import jax.numpy as jnp
import jax

from pbo.networks.jax.q import BaseQFunction


class BasePBOFunction:
    def __init__(self, network: hk.Module, network_key: int, gamma: float, q_function: BaseQFunction) -> None:
        self.gamma = gamma
        self.q_function = q_function

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, weight=jnp.zeros((q_function.q_weights_dimensions)))

        self.loss_and_grad_loss = jax.jit(jax.value_and_grad(self.loss))

    def compute_target(self, batch: dict, weights: jnp.ndarray) -> jnp.ndarray:
        q_params = self.q_function.convert_to_params(weights)
        return batch["reward"].reshape(-1, 1) + self.gamma * self.q_function.max_value(q_params, batch["next_state"])

    def loss(self, pbo_params: hk.Params, batch: dict, weights: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        iterated_weights = self.network.apply(pbo_params, weights)
        iterated_q_params = self.q_function.convert_to_params(iterated_weights)

        loss = jnp.linalg.norm(
            self.q_function.network.apply(iterated_q_params, batch["state"], batch["action"]) - target
        )

        return loss

    def get_fixed_point(self) -> jnp.ndarray:
        raise NotImplementedError


class LinearPBONet(hk.Module):
    def __init__(self, layer_dimension: int) -> None:
        super().__init__(name="LinearNet")
        self.layer_dimension = layer_dimension

    def __call__(self, weight: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.layer_dimension, name="linear")(weight)

        return x


class LinearPBOFunction(BasePBOFunction):
    def __init__(self, network_key: int, gamma: float, q_function: BaseQFunction) -> None:
        def network(weight: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q_function.q_weights_dimensions)(weight)

        super(LinearPBOFunction, self).__init__(network, network_key, gamma, q_function)

    def get_fixed_point(self) -> jnp.ndarray:
        return (
            -jnp.linalg.inv(self.params["LinearNet/linear"]["w"] - jnp.eye(self.q_function.q_weights_dimensions))
            @ self.params["LinearNet/linear"]["b"]
        )
