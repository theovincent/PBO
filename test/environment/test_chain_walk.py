import unittest
import jax
import jax.numpy as jnp
import numpy as np

from pbo.environment.chain_walk import ChainWalkEnv


class TestChainWalkEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.env_key = jax.random.PRNGKey(np.random.randint(0, 100))
        self.n_states = np.random.randint(3, 100)
        self.sucess_probability = np.random.random()
        self.gamma = np.random.random()

    def test_reset(self) -> None:
        env = ChainWalkEnv(self.env_key, self.n_states, self.sucess_probability, self.gamma)

        state = env.reset()
        self.assertAlmostEqual(state, env.state)
        self.assertEqual(state.shape, (1,))
        self.assertGreaterEqual(state[0], 0)
        self.assertLess(state[0], self.n_states)

    def test_step(self) -> None:
        env = ChainWalkEnv(self.env_key, self.n_states, self.sucess_probability, self.gamma)
        state = env.reset()

        for _ in range(10):
            next_state, reward, absorbing, _ = env.step(jnp.array([0]))

            self.assertTrue(next_state[0] == state[0] - 1 or next_state[0] == state[0])
            self.assertTrue(
                (reward[0] == 0 and state[0] > 0 and state[0] < self.n_states)
                or (reward[0] == 1 and (state[0] == 0 or state[0] == self.n_states))
            )
            self.assertFalse(absorbing[0])

            state = next_state

        for _ in range(10):
            next_state, reward, absorbing, _ = env.step(jnp.array([1]))

            self.assertTrue(next_state[0] == state[0] + 1 or next_state[0] == state[0])
            self.assertTrue(
                (reward[0] == 0 and state[0] > 0 and state[0] < self.n_states)
                or (reward[0] == 1 and (state[0] == 0 or state[0] == self.n_states))
            )
            self.assertFalse(absorbing[0])

            state = next_state
