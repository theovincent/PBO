# This file was inspired by https://github.com/MushroomRL/mushroom-rl

from functools import partial

import scipy.linalg as sc_linalg
import jax
import jax.numpy as jnp


class LinearQuadraticEnv:
    """
    This class implements a Linear-Quadratic Regulator.
    This task aims to minimize the undesired deviations from nominal values of
    some controller settings in control problems.
    The system equations in this task are:

    .. math::
        s_{t+1} = As_t + Ba_t

    where s is the state and a is the control signal.

    The reward function is given by:

    .. math::
        r_t = \\left( s_t^TQs_t + a_t^TRa_t + 2 s_t^TSa_t \\right)

    "Policy gradient approaches for multi-objective sequential decision making".
    Parisi S., Pirotta M., Smacchia N., Bascetta L., Restelli M.. 2014

    """

    def __init__(self, env_key: jax.random.PRNGKeyArray, max_init_state: float = None) -> None:
        """
        Constructor.

            Args:
                env_key (int): key to generate the random parameters;
                max_init_state (float, None): start from a random state
                within -max_init_state, max_init_state.

        """
        # Generate a controllable environmnent
        controllable = False

        while not controllable:
            env_key, key = jax.random.split(env_key)
            self.A = jax.random.uniform(key, minval=-1, maxval=1)
            env_key, key = jax.random.split(env_key)
            self.B = jax.random.uniform(key, minval=-1, maxval=1)
            env_key, key = jax.random.split(env_key)
            self.Q = jax.random.uniform(key, minval=-1, maxval=0)
            env_key, key = jax.random.split(env_key)
            self.R = jax.random.uniform(key, minval=-1, maxval=1)
            env_key, key = jax.random.split(env_key)
            self.S = jax.random.uniform(key, minval=-1, maxval=1)

            self.P = sc_linalg.solve_discrete_are(self.A, self.B, self.Q, self.R, s=self.S)[0, 0]

            riccati_respected = self.check_riccati_equation(self.P, self.A, self.B, self.Q, self.R, self.S)
            self.R_hat = self.R + self.B * self.P * self.B

            if self.R_hat < 0 and riccati_respected:
                controllable = True
                self.S_hat = self.S + self.A * self.P * self.B
                self.K = self.S_hat / self.R_hat

                print("Transition: s' = As + Ba")
                print(f"Transition: s' = {self.A}s + {self.B}a")
                print("Reward: Qs² + Ra² + 2 Ssa")
                print(f"Reward: {self.Q}s² + {self.R}a² + {2 * self.S}sa")

        self.max_init_state = max_init_state

        self.optimal_weights = jnp.array(
            [
                self.Q + self.A**2 * self.P,
                self.S + self.A * self.B * self.P,
                self.R + self.B**2 * self.P,
            ]
        )
        self.optimal_bias = jnp.array([self.Q, self.S, self.R])
        self.optimal_slope = jnp.array([self.A**2, self.A * self.B, self.B**2])

    @staticmethod
    def check_riccati_equation(P: float, A: float, B: float, Q: float, R: float, S: float) -> bool:
        return abs(Q + A**2 * P - (S + A * P * B) ** 2 / (R + B**2 * P) - P) < 1e-8

    def reset(self, state: jnp.ndarray) -> jnp.ndarray:
        self.state = state

        return self.state

    def step(self, action: jnp.ndarray) -> tuple:
        reward = self.Q * self.state**2 + self.R * action**2 + 2 * self.S * self.state * action
        self.state = self.A * self.state + self.B * action

        absorbing = False

        return self.state, reward, jnp.array([absorbing]), {}

    def greedy_V(self, weights: jnp.ndarray) -> jnp.ndarray:
        ratio = weights[..., 1] / (weights[..., 2] + 1e-32)
        return (self.Q - 2 * self.S * ratio + self.R * ratio**2) / (1 - (self.A - self.B * ratio) ** 2)
