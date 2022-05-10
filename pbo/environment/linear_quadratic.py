import numpy as np
import scipy.linalg as sc_linalg

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces


class LinearQuadraticEnv(Environment):
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
        max_pos=np.inf,
        max_action=np.inf,
        random_init=False,
        episodic=False,
        gamma=0.9,
        horizon=50,
        initial_state=None,
    ):
        """
        Constructor.

            Args:
                A (np.ndarray): the state dynamics matrix;
                B (np.ndarray): the action dynamics matrix;
                Q (np.ndarray): reward weight matrix for state;
                R (np.ndarray): reward weight matrix for action;
                S (np.ndarray): cross weight matrix;
                max_pos (float, np.inf): maximum value of the state;
                max_action (float, np.inf): maximum value of the action;
                random_init (bool, False): start from a random state;
                episodic (bool, False): end the episode when the state goes over
                the threshold;
                gamma (float, 0.9): discount factor;
                horizon (int, 50): horizon of the mdp.

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

        self._max_pos = max_pos
        self._max_action = max_action
        self._episodic = episodic
        self.random_init = random_init

        self._initial_state = initial_state

        # MDP properties
        high_x = self._max_pos * np.ones(self.A.shape[0])
        low_x = -high_x

        high_u = self._max_action * np.ones(self.B.shape[1])
        low_u = -high_u

        observation_space = spaces.Box(low=low_x, high=high_x)
        action_space = spaces.Box(low=low_u, high=high_u)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            if self.random_init:
                self._state = self._bound(
                    np.random.uniform(-3, 3, size=self.A.shape[0]),
                    self.info.observation_space.low,
                    self.info.observation_space.high,
                )
            elif self._initial_state is not None:
                self._state = self._initial_state
            else:
                init_value = 0.9 * self._max_pos if np.isfinite(self._max_pos) else 10
                self._state = init_value * np.ones(self.A.shape[0])
        else:
            self._state = state

        return self._state

    def step(self, action):
        s = self._state
        a = self._bound(action, self.info.action_space.low, self.info.action_space.high)

        reward = s.T @ self.Q @ s + a.T @ self.R @ a + 2 * s.T @ self.S @ a
        self._state = self.A @ s + self.B @ a

        absorbing = False

        if np.any(np.abs(self._state) > self._max_pos):
            if self._episodic:
                reward = -self._max_pos**2 * 10
                absorbing = True
            else:
                self._state = self._bound(
                    self._state, self.info.observation_space.low, self.info.observation_space.high
                )

        return self._state, reward, absorbing, {}

    def optimal_action(self):
        return -self.K @ self._state
