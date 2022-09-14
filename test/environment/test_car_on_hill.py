import unittest
import jax.numpy as jnp
import numpy as np

from pbo.environment.car_on_hill import CarOnHillEnv


class TestCarOnHillEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gamma = np.random.random()

    def test_reset(self) -> None:
        env = CarOnHillEnv(self.gamma)

        state = env.reset()
        self.assertAlmostEqual(state[0], env.state[0])
        self.assertAlmostEqual(state[1], env.state[1])
        self.assertEqual(state.shape, (2,))
        self.assertAlmostEqual(state[0], -0.5)
        self.assertAlmostEqual(state[1], 0)

    def test_step(self) -> None:
        env = CarOnHillEnv(self.gamma)
        state = env.reset()

        for _ in range(20):
            next_state, reward, absorbing, _ = env.step(jnp.array([np.random.choice([-1, 1])]))

            if not absorbing[0]:
                self.assertTrue(abs(next_state[0]) <= env._max_position)
                self.assertTrue(abs(next_state[1]) <= env._max_velocity)
                self.assertEqual(reward[0], 0)

                state = next_state
            else:
                self.assertTrue(abs(state[0]) > env._max_position or abs(state[1]) > env._max_velocity)
                if state[0] > env._max_position and abs(state[1]) <= env._max_velocity:
                    self.assertEqual(reward[0], 1)
                else:
                    self.assertEqual(reward[0], -1)

                state = env.reset(np.random.random(size=2))
