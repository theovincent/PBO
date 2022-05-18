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

    def __init__(
        self,
        env_key: int,
        initial_state: jnp.ndarray = None,
        max_init_state: float = None,
    ) -> None:
        """
        Constructor.

            Args:
                env_key (int): key to generate the random parameters;
                initial_state (jnp.ndarray, None): start from the given state, if None start
                from a random state;
                max_init_state (float, None): if initial_state is not None, start from a random state
                within -max_init_state, max_init_state.

        """
        assert (
            initial_state is not None or max_init_state is not None
        ), "Either initial_state or max_init_state has to be defined"
        assert (
            initial_state is None or max_init_state is None
        ), "initial_state and max_init_state can't be defined at the same time"

        self.parameters_key, self.reset_key = jax.random.split(env_key)

        # Generate a controllable environmnent
        controllable = False

        while not controllable:
            sekeylf.parameters_key, key = jax.random.split(self.parameters_key)
            self.B = jax.random.uniform(key, (1, 1), minval=-1, maxval=1)
            self.parameters_key, key = jax.random.split(self.parameters_key)
            self.Q = jax.random.uniform(key, (1, 1), minval=-1, maxval=0)
            self.parameters_key, key = jax.random.split(self.parameters_key)
            self.R = jax.random.uniform(key, (1, 1), minval=-1, maxval=1)
            self.parameters_key, key = jax.random.split(self.parameters_key)
            self.S = jax.random.uniform(key, (1, 1), minval=-0.5, maxval=0.5)

            self.P = jnp.array(sc_linalg.solve_discrete_are(self.A, self.B, self.Q, self.R, s=self.S))
            self.R_hat = self.R + self.B.T @ self.P @ self.B

            if self.R_hat < 0:
                controllable = True
                self.S_hat = self.S + self.A.T @ self.P @ self.B
                self.K = sc_linalg.inv(self.R_hat) @ self.S_hat.T

                print("Transition: s' = As + Ba")
                print(f"Transition: s' = {jnp.around(self.A[0, 0], 2)}s + {jnp.around(self.B[0, 0], 2)}a")
                print("Reward: Qs² + Ra² + 2 Ssa")
                print(
                    f"Reward: {jnp.around(self.Q[0, 0], 2)}s² + {jnp.around(self.R[0, 0], 2)}a² + {jnp.around(2 * self.S[0, 0], 2)}sa"
                )

        self.initial_state = initial_state
        self.max_init_state = max_init_state

    def reset(self, state: jnp.ndarray = None) -> jnp.ndarray:
        if state is None:
            if self.initial_state is not None:
                self.state = self.initial_state
            else:
                self.state = jax.random.uniform(
                    self.reset_key, [self.A.shape[0]], minval=-self.max_init_state, maxval=self.max_init_state
                )
        else:
            self.state = state

        return self.state

    def step(self, action: jnp.ndarray) -> tuple:
        reward = self.state.T @ self.Q @ self.state + action.T @ self.R @ action + 2 * self.state.T @ self.S @ action
        self.state = self.A @ self.state + self.B @ action

        absorbing = False

        return self.state, jnp.array([reward]), absorbing, {}

    def optimal_action(self) -> jnp.ndarray:
        return -self.K @ self.state
