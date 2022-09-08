import unittest
import jax
import numpy as np

from pbo.environment.chain_walk import ChainWalkEnv


class TestChainWalkEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.env_key = jax.random.PRNGKey(np.random.randint(0, 100))
        self.n_states = np.random.randint(3, 100)
        self.sucess_probability = np.random.random()
        self.gamma = np.random.random()

    def test_reset(self):
        env = ChainWalkEnv(self.env_key, self.n_states, self.sucess_probability, self.gamma, initial_state=None)

        state = env.reset()
        self.assertAlmostEqual(state, env.state)
        self.assertEqual(state.shape, (1,))
        self.assertGreaterEqual(state[0], 0)
        self.assertGreater(self.n_states, state[0])
