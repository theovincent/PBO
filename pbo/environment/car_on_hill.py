import numpy as np
import jax.numpy as jnp
from scipy.integrate import odeint
from tqdm.notebook import tqdm

from pbo.environment.viewer import Viewer
from pbo.networks.base_q import BaseQ


class CarOnHillEnv:
    """
    The Car On Hill environment as presented in:
    "Tree-Based Batch Mode Reinforcement Learning". Ernst D. et al.. 2005.
    """

    def __init__(self, max_pos: float, max_velocity: float, gamma: float) -> None:
        self.gamma = gamma
        self.max_pos = max_pos
        self.max_velocity = max_velocity
        self.actions_values = [-4.0, 4.0]
        self._g = 9.81
        self._m = 1.0
        self._dt = 0.1

        # Visualization
        self._viewer = Viewer(1, 1)

    def reset(self, state: jnp.ndarray = None) -> jnp.ndarray:
        if state is None:
            self.state = jnp.array([-0.5, 0])
        else:
            self.state = state

        return self.state

    def step(self, action: jnp.ndarray) -> tuple:
        action = self.actions_values[action[0]]
        sa = np.append(self.state, action)
        new_state = odeint(self._dpds, sa, [0, self._dt])

        self.state = jnp.array(new_state[-1, :-1])

        if self.state[0] < -self.max_pos or np.abs(self.state[1]) > self.max_velocity:
            reward = -1.0
            absorbing = True
        elif self.state[0] > self.max_pos and np.abs(self.state[1]) <= self.max_velocity:
            reward = 1.0
            absorbing = True
        else:
            reward = 0.0
            absorbing = False

        return self.state, jnp.array([reward]), absorbing, {}

    def render(self):
        # Slope
        self._viewer.function(0, 1, self._height)

        # Car
        car_body = [
            [-3e-2, 0],
            [-3e-2, 2e-2],
            [-2e-2, 2e-2],
            [-1e-2, 3e-2],
            [1e-2, 3e-2],
            [2e-2, 2e-2],
            [3e-2, 2e-2],
            [3e-2, 0],
        ]

        x_car = (np.array(self.state)[0] + 1) / 2
        y_car = self._height(x_car)
        c_car = [x_car, y_car]
        angle = self._angle(x_car)
        self._viewer.polygon(c_car, angle, car_body, color=(32, 193, 54))

        self._viewer.display(self._dt)

    @staticmethod
    def _angle(x):
        if x < 0.5:
            m = 4 * x - 1
        else:
            m = 1 / ((20 * x**2 - 20 * x + 6) ** 1.5)

        return np.arctan(m)

    @staticmethod
    def _height(x):
        y_neg = 4 * x**2 - 2 * x
        y_pos = (2 * x - 1) / np.sqrt(5 * (2 * x - 1) ** 2 + 1)
        y = np.zeros_like(x)

        mask = x < 0.5
        neg_mask = np.logical_not(mask)
        y[mask] = y_neg[mask]
        y[neg_mask] = y_pos[neg_mask]

        y_norm = (y + 1) / 2

        return y_norm

    def _dpds(self, state_action, t):
        position = state_action[0]
        velocity = state_action[1]
        u = state_action[-1]

        if position < 0.0:
            diff_hill = 2 * position + 1
            diff_2_hill = 2
        else:
            diff_hill = 1 / ((1 + 5 * position**2) ** 1.5)
            diff_2_hill = (-15 * position) / ((1 + 5 * position**2) ** 2.5)

        dp = velocity
        ds = (u - self._g * self._m * diff_hill - velocity**2 * self._m * diff_hill * diff_2_hill) / (
            self._m * (1 + diff_hill**2)
        )

        return dp, ds, 0.0

    def close(self):
        return self._viewer.close()

    def optimal_steps_to_absorbing(self, state: jnp.ndarray, max_steps: int):
        current_states = [state]
        step = 0

        while len(current_states) > 0 and step < max_steps:
            next_states = []
            for state_ in current_states:
                for action in range(2):
                    self.state = state_
                    next_state, reward, _, _ = self.step(jnp.array([action]))

                    if reward == 1:
                        return True, step + 1
                    elif reward == 0:
                        next_states.append(next_state)

            step += 1
            current_states = next_states

        return False, step

    def optimal_v(self, state: jnp.ndarray, max_steps: int) -> float:
        success, step_to_absorbing = self.optimal_steps_to_absorbing([state], max_steps)

        return self.gamma ** (step_to_absorbing - 1) if success else -self.gamma ** (step_to_absorbing - 1)

    def optimal_vs(self, states: jnp.ndarray, max_steps: int) -> jnp.ndarray:
        vs = []

        for state in tqdm(states):
            vs.append(self.optimal_v(state, max_steps))

        return jnp.array(vs)

    def optimal_v_mesh(self, states_x: jnp.ndarray, states_v: jnp.ndarray, max_steps: int) -> jnp.ndarray:
        states_x_mesh, states_v_mesh = jnp.meshgrid(states_x, states_v, indexing="ij")

        states = jnp.hstack((states_x_mesh.reshape((-1, 1)), states_v_mesh.reshape((-1, 1))))

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        return self.optimal_vs(states, max_steps).reshape((states_x.shape[0], states_v.shape[0]))

    def simulate(self, q: BaseQ, horizon: int, initial_state: jnp.ndarray) -> bool:
        self.reset(initial_state)
        absorbing = False
        step = 0

        while not absorbing and step < horizon:
            print(self.state)
            print(q(q.params, self.state, jnp.array([0])))
            print(q(q.params, self.state, jnp.array([1])))
            print()
            if q(q.params, self.state, jnp.array([0])) > q(q.params, self.state, jnp.array([1])):
                _, reward, absorbing, _ = self.step(jnp.array([0]))
            else:
                _, reward, absorbing, _ = self.step(jnp.array([1]))

            step += 1
            self.render()

        self.close()

        return reward == 1
