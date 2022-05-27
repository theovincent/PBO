from functools import partial

import haiku as hk
import jax.numpy as jnp
import jax
import optax

from pbo.networks.q import BaseQ


class BasePBO:
    def __init__(
        self,
        network: hk.Module,
        network_key: int,
        q: BaseQ,
        learning_rate: float,
        max_bellman_iterations: int,
    ) -> None:
        self.q = q

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, weights=jnp.zeros((q.weights_dimension)))

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

        self.iteration_loss_and_grad = []
        for n_iterations in range(max_bellman_iterations + 1):
            self.iteration_loss_and_grad.append(
                jax.jit(jax.value_and_grad(partial(self.iteration_loss, n_iterations=n_iterations)))
            )

        learning_rate_schedule = optax.linear_schedule(
            learning_rate["first"], learning_rate["last"], learning_rate["duration"]
        )
        self.optimizer = optax.adam(learning_rate_schedule)
        self.optimizer_state = self.optimizer.init(self.params)

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, batch_samples: dict, batch_weights: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda weights: batch_samples["reward"]
            + self.q.max_value(self.q.to_params(weights), batch_samples["next_state"])
        )(batch_weights)

    def loss(
        self, pbo_params: hk.Params, sample: dict, batch_weights: jnp.ndarray, batch_targets: jnp.ndarray
    ) -> jnp.ndarray:
        batch_iterated_weights = self.network.apply(pbo_params, batch_weights)

        q_values = jax.vmap(
            lambda weights: self.q.network.apply(self.q.to_params(weights), sample["state"], sample["action"])
        )(batch_iterated_weights)

        return jnp.linalg.norm(q_values - batch_targets)

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: hk.Params, optimizer_state: tuple, batch_samples: jnp.ndarray, batch_weights: jnp.ndarray
    ) -> tuple:
        batch_targets = self.compute_target(batch_samples, batch_weights)

        loss, grad_loss = self.loss_and_grad(params, batch_samples, batch_weights, batch_targets)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def iteration_loss(
        self, pbo_params: hk.Params, sample: dict, batch_weights: jnp.ndarray, n_iterations: int
    ) -> jnp.ndarray:
        batch_iterated_weights = batch_weights
        for _ in jnp.arange(n_iterations):
            batch_iterated_weights = self.network.apply(pbo_params, batch_iterated_weights)
        batch_targets = self.compute_target(sample, batch_iterated_weights)
        batch_targets = jax.lax.stop_gradient(batch_targets)

        batch_iterated_again_weights = self.network.apply(pbo_params, batch_iterated_weights)

        q_values = jax.vmap(
            lambda weights: self.q.network.apply(self.q.to_params(weights), sample["state"], sample["action"])
        )(batch_iterated_again_weights)

        return jnp.linalg.norm(q_values - batch_targets)

    @partial(jax.jit, static_argnames=("self", "n_iterations"))
    def learn_iterations_on_batch(
        self,
        params: hk.Params,
        optimizer_state: tuple,
        batch_samples: jnp.ndarray,
        batch_weights: jnp.ndarray,
        n_iterations: int,
    ) -> tuple:
        loss, grad_loss = self.iteration_loss_and_grad[n_iterations](params, batch_samples, batch_weights)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def fix_point(self) -> jnp.ndarray:
        raise NotImplementedError


class LinearPBONet(hk.Module):
    def __init__(self, layer_dimension: int) -> None:
        super().__init__(name="LinearPBONet")
        self.layer_dimension = layer_dimension

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.layer_dimension, name="linear")(weights)

        return x


class LinearPBO(BasePBO):
    def __init__(self, network_key: int, q: BaseQ, learning_rate: float, max_bellman_iterations: int) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q.weights_dimension)(weights)

        super(LinearPBO, self).__init__(network, network_key, q, learning_rate, max_bellman_iterations)

    def fix_point(self) -> jnp.ndarray:
        return self.params["LinearPBONet/linear"]["b"] @ jnp.linalg.inv(
            jnp.eye(self.q.weights_dimension) - self.params["LinearPBONet/linear"]["w"]
        )

    def contracting(self) -> float:
        return jnp.linalg.norm(self.params["LinearPBONet/linear"]["w"], ord=1)


class OptimalPBO:
    def __init__(self, A: float, B: float, Q: float, R: float, S: float) -> None:
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.S = S

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        # a batch of weights comes with shape (b_s, 3)
        # estimated_p is of shape (b_s)
        estimated_p = weights.T[0] - weights.T[1] ** 2 / weights.T[2]

        return jnp.array(
            [
                self.Q + self.A**2 * estimated_p,
                self.S + self.A * self.B * estimated_p,
                self.R + self.B**2 * estimated_p,
            ]
        ).T


class BaseOptimalPBO:
    def __init__(
        self,
        network: hk.Module,
        network_key: int,
        q_weights_dimension: int,
        learning_rate: float,
        pbo_optimal: OptimalPBO,
        max_bellman_iterations: int,
    ) -> None:
        self.q_weights_dimension = q_weights_dimension
        self.pbo_optimal = pbo_optimal
        self.max_bellman_iterations = max_bellman_iterations

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, weights=jnp.zeros((q_weights_dimension)))

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

        learning_rate_schedule = optax.linear_schedule(
            learning_rate["first"], learning_rate["last"], learning_rate["duration"]
        )
        self.optimizer = optax.sgd(learning_rate_schedule)
        self.optimizer_state = self.optimizer.init(self.params)

    def loss(self, pbo_params: hk.Params, batch_weights: jnp.ndarray) -> jnp.ndarray:
        loss = 0

        batch_weights_iterated = batch_weights
        batch_weights_optimal_iterated = batch_weights

        for _ in jnp.arange(self.max_bellman_iterations):
            batch_weights_iterated = self.network.apply(pbo_params, batch_weights_iterated)
            batch_weights_optimal_iterated = self.pbo_optimal(batch_weights_optimal_iterated)
            batch_weights_optimal_iterated = jax.lax.stop_gradient(batch_weights_optimal_iterated)

            loss += jnp.linalg.norm(batch_weights_optimal_iterated - batch_weights_iterated)

        return loss

    @partial(jax.jit, static_argnames=("self", "n_iterations"))
    def learn_on_batch(self, params: hk.Params, optimizer_state: tuple, batch_weights: jnp.ndarray) -> tuple:
        loss, grad_loss = self.loss_and_grad(params, batch_weights)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss, grad_loss


class OptimalLinearPBO(BaseOptimalPBO):
    def __init__(
        self,
        network_key: int,
        q_weights_dimension: int,
        learning_rate: float,
        pbo_optimal: OptimalPBO,
        max_bellman_iterations: int,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q_weights_dimension)(weights)

        super(OptimalLinearPBO, self).__init__(
            network, network_key, q_weights_dimension, learning_rate, pbo_optimal, max_bellman_iterations
        )

    def fix_point(self) -> jnp.ndarray:
        return self.params["LinearPBONet/linear"]["b"] @ jnp.linalg.inv(
            jnp.eye(self.q_weights_dimension) - self.params["LinearPBONet/linear"]["w"]
        )


from sklearn.linear_model import LinearRegression


class OptimalLinearPBOSKlearn:
    def __init__(self, weights: jnp.ndarray, optimal_iterations: jnp.ndarray) -> None:
        self.fitted_regressor = LinearRegression().fit(weights, optimal_iterations)

        self.params = {
            "OptimalLinearPBONet/linear": {"w": self.fitted_regressor.coef_.T, "b": self.fitted_regressor.intercept_}
        }

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        return weights @ self.params["OptimalLinearPBONet/linear"]["w"] + self.params["OptimalLinearPBONet/linear"]["b"]

    def fix_point(self) -> jnp.ndarray:
        return self.params["OptimalLinearPBONet/linear"]["b"] @ jnp.linalg.inv(
            jnp.eye(3) - self.params["OptimalLinearPBONet/linear"]["w"]
        )
