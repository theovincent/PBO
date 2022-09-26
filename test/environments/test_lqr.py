import unittest
import jax
import jax.numpy as jnp
import numpy as np

from pbo.environments.linear_quadratic import LinearQuadraticEnv


class TestLinearQuadraticEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.env_key = jax.random.PRNGKey(np.random.randint(0, 100))
        self.max_init_state = np.random.random() * 10

    def test_reset(self) -> None:
        env = LinearQuadraticEnv(self.env_key, self.max_init_state)

        state = env.reset()
        self.assertAlmostEqual(state, env.state)
        self.assertEqual(state.shape, (1,))
        self.assertGreaterEqual(state[0], -self.max_init_state)
        self.assertLess(state[0], self.max_init_state)

    def test_step(self) -> None:
        env = LinearQuadraticEnv(self.env_key, self.max_init_state)
        state = env.reset()

        for _ in range(20):
            action = jnp.array([np.random.random() * 10])
            next_state, reward, absorbing, _ = env.step(action)

            self.assertEqual(next_state.shape, (1,))
            self.assertEqual(reward.shape, (1,))
            self.assertEqual(absorbing.shape, (1,))
            self.assertAlmostEqual(next_state[0], env.A * state[0] + env.B * action[0])
            self.assertAlmostEqual(
                reward[0], env.Q * state[0] ** 2 + env.R * action[0] ** 2 + 2 * env.S * state[0] * action[0]
            )
            self.assertFalse(absorbing[0])

            state = next_state
