import unittest
import jax
import jax.numpy as jnp
import numpy as np

from pbo.networks.learnable_q import FullyConnectedQ


class TestTableQ(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state_dim = np.random.randint(1, 5)
        self.action_dim = np.random.randint(1, 5)
        self.actions_on_max = np.random.randint(-10, 10, size=(np.random.randint(1, 30), self.action_dim))
        self.gamma = np.random.random()
        self.network_key = jax.random.PRNGKey(np.random.randint(0, 100))
        self.layers_dimension = np.random.randint(1, 50, size=np.random.randint(1, 4))
        self.initial_weight_std = np.abs(np.random.random() * 10)

    def test_zero_initializer(self) -> None:
        q = FullyConnectedQ(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            actions_on_max=self.actions_on_max,
            gamma=self.gamma,
            network_key=self.network_key,
            layers_dimension=self.layers_dimension,
            initial_weight_std=self.initial_weight_std,
            zero_initializer=True,
        )

        states = np.random.random(size=(20, self.state_dim)) * 100
        actions = np.random.random(size=(20, self.action_dim)) * 100
        self.assertAlmostEqual(jnp.linalg.norm(q(q.params, states, actions)), 0)

        q = FullyConnectedQ(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            actions_on_max=self.actions_on_max,
            gamma=self.gamma,
            network_key=self.network_key,
            layers_dimension=self.layers_dimension,
            initial_weight_std=self.initial_weight_std,
            zero_initializer=False,
        )

        self.assertGreater(jnp.linalg.norm(q(q.params, states, actions)), 0)
        self.assertGreater(jnp.linalg.norm(q.to_weights(q.params)), 0)

    # def test_weights_dimension(self) -> None:
    #     q = LQRQ(self.n_actions_on_max, self.max_action_on_max, self.network_key, zero_initializer=False)

    #     weights = q.to_weights(q.params)

    #     self.assertEqual(weights.shape[0], q.weights_dimension)

    # def test_to_params_to_weights(self) -> None:
    #     q = LQRQ(self.n_actions_on_max, self.max_action_on_max, self.network_key, zero_initializer=False)

    #     for _ in range(20):
    #         weights = np.random.random(size=q.weights_dimension)

    #         computed_weights = q.to_weights(q.to_params(weights))

    #         self.assertAlmostEqual(jnp.linalg.norm(computed_weights - weights), 0)

    # def test_max_value(self) -> None:
    #     q = LQRQ(self.n_actions_on_max, self.max_action_on_max, self.network_key, zero_initializer=False)
    #     actions_on_max = np.linspace(-self.max_action_on_max, self.max_action_on_max, self.n_actions_on_max).reshape(
    #         (self.n_actions_on_max, 1)
    #     )

    #     for batch_size in range(20):
    #         params = q.to_params(np.random.random(size=q.weights_dimension))
    #         states = np.random.random(size=(batch_size, 1))

    #         max_values = np.zeros(batch_size)
    #         for idx_state, state in enumerate(states):
    #             max_values[idx_state] = q(params, state, actions_on_max).max()

    #         max_computed_values = q.max_value(params, states)

    #         self.assertAlmostEqual(
    #             jnp.linalg.norm(max_computed_values - max_values.reshape((batch_size, 1))), 0, places=4
    #         )

    # def test_compute_target(self) -> None:
    #     q = LQRQ(self.n_actions_on_max, self.max_action_on_max, self.network_key, zero_initializer=False)

    #     for batch_size in range(1, 21):
    #         weights = np.random.random(size=q.weights_dimension)
    #         rewards = np.random.random(size=batch_size)
    #         absorbings = np.random.randint(0, 2, size=batch_size, dtype=bool)
    #         next_states = np.random.random(size=(batch_size, 1))
    #         samples = {
    #             "reward": rewards.reshape((batch_size, 1)),
    #             "absorbing": absorbings.reshape((batch_size, 1)),
    #             "next_state": next_states.reshape((batch_size, 1)),
    #         }

    #         target = rewards + (1 - absorbings) * q.gamma * q.max_value(q.to_params(weights), next_states).reshape(
    #             (batch_size)
    #         )

    #         computed_target = q.compute_target(weights.reshape((1, q.weights_dimension)), samples)

    #         self.assertAlmostEqual(jnp.linalg.norm(target.reshape((1, batch_size, 1)) - computed_target), 0, places=6)
