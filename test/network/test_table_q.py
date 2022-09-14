import unittest
import jax
import jax.numpy as jnp
import numpy as np

from pbo.networks.learnable_q import TableQ


class TestTableQ(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_states = np.random.randint(3, 100)
        self.n_actions = np.random.randint(2, 10)
        self.gamma = np.random.random()
        self.network_key = jax.random.PRNGKey(np.random.randint(0, 100))

    def test_zero_initializer(self) -> None:
        q = TableQ(self.n_states, self.n_actions, self.gamma, self.network_key, zero_initializer=True)

        states = np.random.randint(0, self.n_states, size=20)
        actions = np.random.randint(0, self.n_actions, size=20)
        self.assertAlmostEqual(jnp.linalg.norm(q(q.params, states, actions)), 0)
        self.assertAlmostEqual(jnp.linalg.norm(q.to_weights(q.params)), 0)

        q = TableQ(self.n_states, self.n_actions, self.gamma, self.network_key, zero_initializer=False)

        self.assertGreater(jnp.linalg.norm(q(q.params, states, actions)), 0)
        self.assertGreater(jnp.linalg.norm(q.to_weights(q.params)), 0)

    def test_weights_dimension(self) -> None:
        q = TableQ(self.n_states, self.n_actions, self.gamma, self.network_key, zero_initializer=False)

        weights = q.to_weights(q.params)

        self.assertEqual(weights.shape[0], q.weights_dimension)

    def test_to_params_to_weights(self) -> None:
        q = TableQ(self.n_states, self.n_actions, self.gamma, self.network_key, zero_initializer=False)

        for _ in range(20):
            weights = np.random.random(size=q.weights_dimension)

            computed_weights = q.to_weights(q.to_params(weights))

            self.assertAlmostEqual(jnp.linalg.norm(computed_weights - weights), 0)

    def test_max_value(self) -> None:
        q = TableQ(self.n_states, self.n_actions, self.gamma, self.network_key, zero_initializer=False)

        for _ in range(20):
            weights = np.random.random(size=q.weights_dimension)
            max_values = weights.reshape((self.n_states, self.n_actions)).max(axis=1, keepdims=True)

            max_computed_values = q.max_value(q.to_params(weights), jnp.arange(self.n_states))

            self.assertAlmostEqual(jnp.linalg.norm(max_computed_values - max_values), 0)

    def test_compute_target(self) -> None:
        q = TableQ(self.n_states, self.n_actions, self.gamma, self.network_key, zero_initializer=False)

        for batch_size in range(1, 21):
            weights = np.random.random(size=q.weights_dimension)
            rewards = np.random.random(size=batch_size)
            absorbings = np.random.randint(0, 2, size=batch_size, dtype=bool)
            next_states = np.random.randint(0, self.n_states, size=batch_size)
            samples = {
                "reward": rewards.reshape((batch_size, 1)),
                "absorbing": absorbings.reshape((batch_size, 1)),
                "next_state": next_states.reshape((batch_size, 1)),
            }

            target = rewards + (1 - absorbings) * q.gamma * q.max_value(q.to_params(weights), next_states).reshape(
                (batch_size)
            )

            computed_target = q.compute_target(weights.reshape((1, q.weights_dimension)), samples)

            self.assertAlmostEqual(jnp.linalg.norm(target.reshape((1, batch_size, 1)) - computed_target), 0, places=6)
