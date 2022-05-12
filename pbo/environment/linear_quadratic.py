import numpy as np
import scipy.linalg as sc_linalg


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
        max_state: float,
        max_action: float,
        initial_state: np.ndarray = None,
        episodic: bool = False,
    ) -> None:
        """
        Constructor.

            Args:
                max_state (float): maximum value of the state;
                max_action (float): maximum value of the action;
                initial_state (np.ndarray, None): start from the given state, if None start
                from a random state;
                episodic (bool, False): end the episode when the state goes over
                the threshold;
        """
        # Generate a controllable environmnent
        controllable = False

        while not controllable:
            self.A = np.random.uniform(-10, 10, size=(1, 1))
            self.B = np.random.uniform(-10, 10, size=(1, 1))
            self.Q = np.random.uniform(-10, 0, size=(1, 1))
            self.R = np.random.uniform(-10, 10, size=(1, 1))
            self.S = np.random.uniform(-5, 5, size=(1, 1))

            self.P = sc_linalg.solve_discrete_are(self.A, self.B, self.Q, self.R, s=self.S)
            self.R_hat = self.R + self.B.T @ self.P @ self.B

            if self.R_hat < 0:
                controllable = True
                self.S_hat = self.S + self.A.T @ self.P @ self.B
                self.K = sc_linalg.inv(self.R_hat) @ self.S_hat.T

                print("Transition: s' = As + Ba")
                print(f"Transition: s' = {np.around(self.A[0, 0], 2)}s + {np.around(self.B[0, 0], 2)}a")
                print("Reward: Qs² + Ra² + 2 Ssa")
                print(
                    f"Reward: {np.around(self.Q[0, 0], 2)}s² + {np.around(self.R[0, 0], 2)}a² + {np.around(2 * self.S[0, 0], 2)}sa"
                )

        self.max_state = max_state
        self.max_action = max_action
        self.initial_state = initial_state
        self.episodic = episodic

    def reset(self, state: np.ndarray = None) -> np.ndarray:
        if state is None:
            if self.initial_state is not None:
                self.state = self.initial_state
            else:
                self.state = np.random.uniform(-self.max_state, self.max_state, size=self.A.shape[0])
        else:
            self.state = state

        return self.state

    def step(self, action: np.ndarray) -> tuple:
        s = self.state
        a = self._bound(action, -self.max_action, self.max_action)

        reward = s.T @ self.Q @ s + a.T @ self.R @ a + 2 * s.T @ self.S @ a
        self.state = self.A @ s + self.B @ a

        absorbing = False

        if np.any(np.abs(self.state) > self.max_state):
            if self.episodic:
                reward = -self.max_state**2 * 10
                absorbing = True
            else:
                self.state = self._bound(self.state, -self.max_state, self.max_state)

        return self.state, reward, absorbing, {}

    def optimal_action(self) -> np.ndarray:
        return self._bound(-self.K @ self.state, -self.max_action, self.max_action)

    @staticmethod
    def _bound(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
        """
        Method used to bound state and action variables.

        Args:
            x: the variable to bound;
            min_value: the minimum value;
            max_value: the maximum value;

        Returns:
            The bounded variable.

        """
        return np.maximum(min_value, np.minimum(x, max_value))
