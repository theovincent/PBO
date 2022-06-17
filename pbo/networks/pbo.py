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
        add_infinity: bool,
        importance_iteration: list,
    ) -> None:
        self.q = q
        self.max_bellman_iterations = max_bellman_iterations
        self.add_infinity = add_infinity
        self.importance_iteration = jnp.array(importance_iteration)

        self.network = hk.without_apply_rng(hk.transform(network))
        self.params = self.network.init(rng=network_key, weights=jnp.zeros((q.weights_dimension)))

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.l2_loss))

        learning_rate_schedule = optax.linear_schedule(
            learning_rate["first"], learning_rate["last"], learning_rate["duration"]
        )
        self.optimizer = optax.adam(learning_rate_schedule)
        self.optimizer_state = self.optimizer.init(self.params)

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, batch_weights: jnp.ndarray, batch_samples: dict) -> jnp.ndarray:
        return jax.vmap(
            lambda weights: batch_samples["reward"]
            + self.q.max_value(self.q.to_params(weights), batch_samples["next_state"])
        )(batch_weights)

    def l2_loss(self, pbo_params: hk.Params, batch_weights: jnp.ndarray, sample: dict) -> float:
        loss = 0
        batch_iterated_weights = batch_weights

        for idx_iteration in jnp.arange(self.max_bellman_iterations):
            batch_targets = self.compute_target(batch_iterated_weights, sample)
            batch_targets = jax.lax.stop_gradient(batch_targets)

            batch_iterated_weights = self.network.apply(pbo_params, batch_iterated_weights)

            q_values = jax.vmap(
                lambda weights: self.q.network.apply(self.q.to_params(weights), sample["state"], sample["action"])
            )(batch_iterated_weights)

            loss += self.importance_iteration[idx_iteration] * jnp.linalg.norm(q_values - batch_targets)

        loss /= self.max_bellman_iterations

        if self.add_infinity:
            fixed_point = self.safe_fixed_point(pbo_params).reshape((1, -1))
            batch_targets = self.compute_target(fixed_point, sample)
            batch_targets = jax.lax.stop_gradient(batch_targets)

            q_values = jax.vmap(
                lambda weights: self.q.network.apply(self.q.to_params(weights), sample["state"], sample["action"])
            )(fixed_point)

            loss += self.importance_iteration[-1] * jnp.linalg.norm(q_values - batch_targets)

        return loss

    @partial(jax.jit, static_argnames="self")
    def l1_loss(self, pbo_params: hk.Params, batch_weights: jnp.ndarray, sample: dict) -> float:
        loss = 0
        batch_iterated_weights = batch_weights

        for idx_iteration in jnp.arange(self.max_bellman_iterations):
            batch_targets = self.compute_target(batch_iterated_weights, sample)

            batch_iterated_weights = self.network.apply(pbo_params, batch_iterated_weights)

            q_values = jax.vmap(
                lambda weights: self.q.network.apply(self.q.to_params(weights), sample["state"], sample["action"])
            )(batch_iterated_weights)

            loss += self.importance_iteration[idx_iteration] * jnp.abs(q_values - batch_targets).mean()

        loss /= self.max_bellman_iterations

        if self.add_infinity:
            fixed_point = self.fixed_point(pbo_params).reshape((1, -1))
            batch_targets = self.compute_target(fixed_point, sample)

            q_values = jax.vmap(
                lambda weights: self.q.network.apply(self.q.to_params(weights), sample["state"], sample["action"])
            )(fixed_point)

            loss += self.importance_iteration[-1] * jnp.abs(q_values - batch_targets).mean()

        return loss

    @partial(jax.jit, static_argnames=("self", "n_discrete_states", "n_discrete_actions"))
    def l1_loss_mesh(
        self,
        pbo_params: hk.Params,
        batch_weights: jnp.ndarray,
        sample: dict,
        n_discrete_states: int,
        n_discrete_actions: int,
    ) -> jnp.ndarray:
        loss_mesh = jnp.zeros((n_discrete_states, n_discrete_actions))
        batch_iterated_weights = batch_weights

        for idx_iteration in jnp.arange(self.max_bellman_iterations):
            batch_targets = self.compute_target(batch_iterated_weights, sample).reshape(
                (-1, n_discrete_states, n_discrete_actions)
            )
            batch_iterated_weights = self.network.apply(pbo_params, batch_iterated_weights)

            q_values = jax.vmap(
                lambda weights: self.q.network.apply(self.q.to_params(weights), sample["state"], sample["action"])
            )(batch_iterated_weights).reshape((-1, n_discrete_states, n_discrete_actions))

            loss_mesh = loss_mesh.at[:].add(
                self.importance_iteration[idx_iteration] * jnp.abs(q_values - batch_targets).mean(axis=0)
            )

        loss_mesh /= self.max_bellman_iterations

        if self.add_infinity:
            fixed_point = self.fixed_point(pbo_params).reshape((1, -1))
            batch_targets = self.compute_target(fixed_point, sample).reshape(
                (-1, n_discrete_states, n_discrete_actions)
            )

            q_values = jax.vmap(
                lambda weights: self.q.network.apply(self.q.to_params(weights), sample["state"], sample["action"])
            )(fixed_point).reshape((-1, n_discrete_states, n_discrete_actions))

            loss_mesh = loss_mesh.at[:].add(
                self.importance_iteration[-1] * jnp.abs(q_values - batch_targets).mean(axis=0)
            )

        return loss_mesh

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: hk.Params, optimizer_state: tuple, batch_weights: jnp.ndarray, batch_samples: jnp.ndarray
    ) -> tuple:
        loss, grad_loss = self.loss_and_grad(params, batch_weights, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        raise NotImplementedError

    def safe_fixed_point(self, params: hk.Params) -> jnp.ndarray:
        raise NotImplementedError


class LinearPBONet(hk.Module):
    def __init__(self, layer_dimension: int) -> None:
        super().__init__(name="LinearPBONet")
        self.layer_dimension = layer_dimension

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        x = hk.Linear(self.layer_dimension, name="linear")(weights)

        return x


class LinearPBO(BasePBO):
    def __init__(
        self,
        network_key: int,
        q: BaseQ,
        learning_rate: float,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q.weights_dimension)(weights)

        super(LinearPBO, self).__init__(
            network, network_key, q, learning_rate, max_bellman_iterations, add_infinity, importance_iteration
        )

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        return params["LinearPBONet/linear"]["b"] @ jnp.linalg.inv(
            jnp.eye(self.q.weights_dimension) - params["LinearPBONet/linear"]["w"]
        )

    def safe_fixed_point(self, params: hk.Params) -> jnp.ndarray:
        return jnp.linalg.solve(
            jnp.eye(self.q.weights_dimension) - params["LinearPBONet/linear"]["w"].T,
            params["LinearPBONet/linear"]["b"].T,
        ).T

    def contracting_factor(self) -> float:
        return jnp.linalg.norm(self.params["LinearPBONet/linear"]["w"], ord=1)


class Linear1ParamPBONet(hk.Module):
    def __init__(self) -> None:
        super().__init__(name="Linear1ParamPBONet")

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        estimated_p = weights[:, 0] - weights[:, 1] ** 2 / weights[:, 2]

        x = hk.Linear(3, name="linear")(estimated_p.reshape((-1, 1)))

        return x


class Linear1ParamPBO(BasePBO):
    def __init__(
        self,
        network_key: int,
        q: BaseQ,
        learning_rate: float,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet()(weights)

        super(Linear1ParamPBO, self).__init__(
            network, network_key, q, learning_rate, max_bellman_iterations, add_infinity, importance_iteration
        )

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        return params["LinearPBONet/linear"]["b"] @ jnp.linalg.inv(
            jnp.eye(self.q.weights_dimension) - params["LinearPBONet/linear"]["w"]
        )

    def safe_fixed_point(self, params: hk.Params) -> jnp.ndarray:
        return jnp.linalg.solve(
            jnp.eye(self.q.weights_dimension) - params["LinearPBONet/linear"]["w"].T,
            params["LinearPBONet/linear"]["b"].T,
        ).T

    def contracting_factor(self) -> float:
        return jnp.linalg.norm(self.params["LinearPBONet/linear"]["w"], ord=1)


class OptimalPBO:
    def __init__(self, A: float, B: float, Q: float, R: float, S: float, P: float, importance_iteration: list) -> None:
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.S = S
        self.P = P
        self.importance_iteration = jnp.array(importance_iteration)

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

    def l1_loss(
        self,
        pbo: BasePBO,
        q: BaseQ,
        batch_weights: jnp.ndarray,
        sample: dict,
        max_bellman_iterations: int,
        add_infinity: bool,
    ) -> float:
        loss = 0
        batch_iterated_weights = batch_weights

        for idx_iteration in jnp.arange(max_bellman_iterations):
            batch_targets = pbo.compute_target(batch_iterated_weights, sample)

            batch_iterated_weights = self(batch_iterated_weights)

            q_values = jax.vmap(
                lambda weights: q.network.apply(q.to_params(weights), sample["state"], sample["action"])
            )(batch_iterated_weights)

            loss += self.importance_iteration[idx_iteration] * jnp.abs(q_values - batch_targets).mean()

        loss / max_bellman_iterations

        if add_infinity:
            fixed_point = self.fixed_point().reshape((1, -1))
            batch_targets = pbo.compute_target(fixed_point, sample)

            q_values = jax.vmap(
                lambda weights: q.network.apply(q.to_params(weights), sample["state"], sample["action"])
            )(fixed_point)

            loss += self.importance_iteration[-1] * jnp.abs(q_values - batch_targets).mean()

        return loss

    def fixed_point(self) -> jnp.ndarray:
        return jnp.array(
            [
                self.Q + self.A**2 * self.P,
                self.S + self.A * self.B * self.P,
                self.R + self.B**2 * self.P,
            ]
        )


class BaseWeightsOptimalPBO:
    def __init__(
        self,
        network: hk.Module,
        network_key: int,
        q_weights_dimension: int,
        learning_rate: float,
        pbo_optimal: OptimalPBO,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
    ) -> None:
        self.q_weights_dimension = q_weights_dimension
        self.pbo_optimal = pbo_optimal
        self.max_bellman_iterations = max_bellman_iterations
        self.add_infinity = add_infinity
        self.importance_iteration = jnp.array(importance_iteration)

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

        for idx_iteration in jnp.arange(self.max_bellman_iterations):
            batch_weights_iterated = self.network.apply(pbo_params, batch_weights_iterated)
            batch_weights_optimal_iterated = self.pbo_optimal(batch_weights_optimal_iterated)
            batch_weights_optimal_iterated = jax.lax.stop_gradient(batch_weights_optimal_iterated)

            loss += self.importance_iteration[idx_iteration] * jnp.linalg.norm(
                batch_weights_optimal_iterated - batch_weights_iterated
            )

        loss /= self.max_bellman_iterations

        if self.add_infinity:
            batch_fixed_point = self.fixed_point(pbo_params)
            loss += self.importance_iteration[-1] * jnp.linalg.norm(batch_fixed_point - self.pbo_optimal.fixed_point())

        return loss

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: hk.Params, optimizer_state: tuple, batch_weights: jnp.ndarray) -> tuple:
        loss, grad_loss = self.loss_and_grad(params, batch_weights)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss, grad_loss

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        raise NotImplementedError

    def safe_fixed_point(self, params: hk.Params) -> jnp.ndarray:
        raise NotImplementedError


class WeightsOptimalLinearPBO(BaseWeightsOptimalPBO):
    def __init__(
        self,
        network_key: int,
        q_weights_dimension: int,
        learning_rate: float,
        pbo_optimal: OptimalPBO,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q_weights_dimension)(weights)

        super(WeightsOptimalLinearPBO, self).__init__(
            network,
            network_key,
            q_weights_dimension,
            learning_rate,
            pbo_optimal,
            max_bellman_iterations,
            add_infinity,
            importance_iteration,
        )

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        return params["LinearPBONet/linear"]["b"] @ jnp.linalg.inv(
            jnp.eye(self.q_weights_dimension) - params["LinearPBONet/linear"]["w"]
        )

    def safe_fixed_point(self, params: hk.Params) -> jnp.ndarray:
        return jnp.linalg.solve(
            jnp.eye(self.q_weights_dimension) - params["LinearPBONet/linear"]["w"].T,
            params["LinearPBONet/linear"]["b"].T,
        ).T


class OptimalTablePBO:
    def __init__(self, q: BaseQ, optimal_bellman_operator, optimal_q: jnp.ndarray, importance_iteration: list) -> None:
        self.q = q
        self.optimal_bellman_operator = optimal_bellman_operator
        self.optimal_q = optimal_q
        self.importance_iteration = jnp.array(importance_iteration)

    def __call__(self, weights: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda weights_: self.optimal_bellman_operator(self.q.to_params(weights_)["TableQNet"]["table"]).flatten()
        )(weights)

    def l1_loss(
        self,
        pbo: BasePBO,
        q: BaseQ,
        batch_weights: jnp.ndarray,
        sample: dict,
        max_bellman_iterations: int,
        add_infinity: bool,
    ) -> float:
        loss = 0
        batch_iterated_weights = batch_weights

        for idx_iteration in jnp.arange(max_bellman_iterations):
            batch_targets = pbo.compute_target(batch_iterated_weights, sample)

            batch_iterated_weights = self(batch_iterated_weights)

            q_values = jax.vmap(
                lambda weights: q.network.apply(q.to_params(weights), sample["state"], sample["action"])
            )(batch_iterated_weights)

            loss += self.importance_iteration[idx_iteration] * jnp.abs(q_values - batch_targets).mean()

        loss / max_bellman_iterations

        if add_infinity:
            fixed_point = self.fixed_point().reshape((1, -1))
            batch_targets = pbo.compute_target(fixed_point, sample)

            q_values = jax.vmap(
                lambda weights: q.network.apply(q.to_params(weights), sample["state"], sample["action"])
            )(fixed_point)

            loss += self.importance_iteration[-1] * jnp.abs(q_values - batch_targets).mean()

        return loss

    def fixed_point(self) -> jnp.ndarray:
        return self.optimal_q


class BaseWeightsOptimalTablePBO:
    def __init__(
        self,
        network: hk.Module,
        network_key: int,
        q_weights_dimension: int,
        learning_rate: float,
        pbo_optimal: OptimalTablePBO,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
    ) -> None:
        self.q_weights_dimension = q_weights_dimension
        self.pbo_optimal = pbo_optimal
        self.max_bellman_iterations = max_bellman_iterations
        self.add_infinity = add_infinity
        self.importance_iteration = jnp.array(importance_iteration)

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

        for idx_iteration in jnp.arange(self.max_bellman_iterations):
            batch_weights_iterated = self.network.apply(pbo_params, batch_weights_iterated)
            batch_weights_optimal_iterated = self.pbo_optimal(batch_weights_optimal_iterated)
            batch_weights_optimal_iterated = jax.lax.stop_gradient(batch_weights_optimal_iterated)

            loss += self.importance_iteration[idx_iteration] * jnp.linalg.norm(
                batch_weights_optimal_iterated - batch_weights_iterated
            )

        loss /= self.max_bellman_iterations

        if self.add_infinity:
            batch_fixed_point = self.fixed_point(pbo_params)
            loss += self.importance_iteration[-1] * jnp.linalg.norm(batch_fixed_point - self.pbo_optimal.fixed_point())

        return loss

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: hk.Params, optimizer_state: tuple, batch_weights: jnp.ndarray) -> tuple:
        loss, grad_loss = self.loss_and_grad(params, batch_weights)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss, grad_loss

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        raise NotImplementedError

    def safe_fixed_point(self, params: hk.Params) -> jnp.ndarray:
        raise NotImplementedError


class WeightsOptimalLinearTablePBO(BaseWeightsOptimalTablePBO):
    def __init__(
        self,
        network_key: int,
        q_weights_dimension: int,
        learning_rate: float,
        pbo_optimal: OptimalTablePBO,
        max_bellman_iterations: int,
        add_infinity: bool,
        importance_iteration: list,
    ) -> None:
        def network(weights: jnp.ndarray) -> jnp.ndarray:
            return LinearPBONet(q_weights_dimension)(weights)

        super(WeightsOptimalLinearTablePBO, self).__init__(
            network,
            network_key,
            q_weights_dimension,
            learning_rate,
            pbo_optimal,
            max_bellman_iterations,
            add_infinity,
            importance_iteration,
        )

    def fixed_point(self, params: hk.Params) -> jnp.ndarray:
        return params["LinearPBONet/linear"]["b"] @ jnp.linalg.inv(
            jnp.eye(self.q_weights_dimension) - params["LinearPBONet/linear"]["w"]
        )

    def safe_fixed_point(self, params: hk.Params) -> jnp.ndarray:
        return jnp.linalg.solve(
            jnp.eye(self.q_weights_dimension) - params["LinearPBONet/linear"]["w"].T,
            params["LinearPBONet/linear"]["b"].T,
        ).T
