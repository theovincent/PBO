import unittest
import jax
import jax.numpy as jnp
import numpy as np

from pbo.environments.bicycle import BicycleEnv


class TestBicycleEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.env_key = jax.random.PRNGKey(np.random.randint(0, 100))
        self.actions = [jnp.array([d, T]) for d in [-1, 0, 1] for T in [-1, 0, 1]]

    def test_reset(self) -> None:
        env = BicycleEnv(self.env_key)

        state = env.reset()
        for idx in range(5):
            self.assertAlmostEqual(state[idx], env.state[idx])
            self.assertAlmostEqual(state[idx], 0)

    def test_step(self) -> None:
        env = BicycleEnv(self.env_key)
        env.reset()

        for _ in range(20):
            next_state, reward, absorbing, _ = env.step(self.actions[np.random.randint(9)])

            self.assertEqual(next_state.shape, (5,))
            self.assertEqual(reward.shape, (1,))
            self.assertEqual(absorbing.shape, (1,))
            self.assertTrue(abs(next_state[2]) <= env.theta_bound)

            if not absorbing[0]:
                self.assertTrue(abs(next_state[0]) <= env.omega_bound)
                self.assertEqual(reward[0], 0)
            else:
                self.assertTrue(abs(next_state[0]) > env.omega_bound)
                self.assertEqual(reward[0], -1)

                env.reset(np.random.random(size=2))
