import jax
import jax.numpy as jnp


class BicycleEnv:
    """
    The Bicycle balancing/riding environment as presented in:
    Learning to Drive a Bicycle using Reinforcement Learning and Shaping.
    Jette Randlov and Preben Alstrom. 1998.
    """

    def __init__(self, env_key: int) -> None:
        self.reset_key, self.noise_key = jax.random.split(env_key)

        self.noise = 0.02
        self.state_bounds = jnp.array(
            [
                [-jnp.pi * 12.0 / 180.0, jnp.pi * 12.0 / 180.0],
                [-jnp.pi * 2.0 / 180.0, jnp.pi * 2.0 / 180.0],
                [-jnp.pi, jnp.pi],
                [-jnp.pi * 80.0 / 180.0, jnp.pi * 80.0 / 180.0],
                [-jnp.pi * 2.0 / 180.0, jnp.pi * 2.0 / 180.0],
            ]
        )
        self.reward_fall = -1.0
        self.reward_goal = 0.01
        self.goal_rsqrd = 100.0  # Square of the radius around the goal (10m)^2
        self.goal_loc = jnp.array([1000.0, 0])

        # Units in Meters and Kilograms
        self._c = 0.66  # Horizontal distance between bottom of front wheel and center of mass
        self._d_cm = 0.30  # Vertical dist between center of mass and the cyclist
        self._h = 0.94  # Height of the center of mass over the ground
        self._l = 1.11  # Distance between front tire and back tire at point on ground
        self._M_c = 15.0  # Mass of bicycle
        self._M_d = 1.7  # Mass of tire
        self._M_p = 60  # Mass of cyclist
        self._r = 0.34  # Radius of tire
        self._v = 10.0 / 3.6  # Velocity of bicycle (converted from km/h to m/s)

        # Useful precomputations
        self._M = self._M_p + self._M_c
        self._Inertia_bc = (13.0 / 3.0) * self._M_c * self._h**2 + self._M_p * (self._h + self._d_cm) ** 2
        self._Inertia_dv = self._M_d * self._r**2
        self._Inertia_dl = 0.5 * self._M_d * self._r**2
        self._sigma_dot = self._v / self._r

        # Simulation Constants
        self._g = 9.8
        self._dt = 0.01  # 0.02
        self._sim_steps = 1  # 10

    def reset(self, state: jnp.ndarray = None, position: jnp.ndarray = None) -> jnp.ndarray:
        if state is None:
            self.state = jnp.zeros((5))
            self.state = self.state.at[-1].set(jax.random.uniform(self.reset_key, minval=-jnp.pi, maxval=jnp.pi))
        else:
            self.state = state

        self.position = jnp.zeros((2))

        return self.state

    def step(self, action: jnp.ndarray) -> jnp.ndarray:
        # action in [-1, 0, 1] x [-1, 0, 1]
        d = 0.02 * action[0]  # Displacement of center of mass (in meters)
        T = 2.0 * action[1]  # Torque on handle bars

        # Add noise to action
        self.noise_key, key = jax.random.split(self.noise_key)
        d += jax.random.uniform(key, minval=-1, maxval=1) * self.noise

        omega, omega_dot, theta, theta_dot, psi = self.state
        x_b, y_b = self.position

        goal_angle_old = self._angle_between(self._goal_loc, jnp.array([x_f - x_b, y_f - y_b])) * jnp.pi / 180.0
        if x_f == x_b and (y_f - y_b) < 0:
            old_psi = jnp.pi
        elif (y_f - y_b) > 0:
            old_psi = jnp.arctan((x_b - x_f) / (y_f - y_b))
        else:
            old_psi = jnp.sign(x_b - x_f) * (jnp.pi / 2.0) - jnp.arctan((y_f - y_b) / (x_b - x_f))

        for step in range(self._sim_steps):
            if theta == 0:  # Infinite radius tends to not be handled well
                r_f = r_b = r_CM = 1.0e8
            else:
                r_f = self._l / jnp.abs(jnp.sin(theta))
                r_b = self._l / jnp.abs(jnp.tan(theta))  # self.l / jnp.abs(jnp.tan(from pyrl.misc import matrixtheta))
                r_CM = jnp.sqrt((self._l - self._c) ** 2 + (self._l**2 / jnp.tan(theta) ** 2))

            varphi = omega + jnp.arctan(d / self._h)

            omega_ddot = self._h * self._M * self._gravity * jnp.sin(varphi)
            omega_ddot -= jnp.cos(varphi) * (
                self._Inertia_dv * self._sigma_dot * theta_dot
                + jnp.sign(theta)
                * self._v**2
                * (self._M_d * self._r * (1.0 / r_f + 1.0 / r_b) + self._M * self._h / r_CM)
            )
            omega_ddot /= self._Inertia_bc

            theta_ddot = (T - self._Inertia_dv * self._sigma_dot * omega_dot) / self._Inertia_dl

            df = self._delta_time / float(self._sim_steps)
            omega_dot += df * omega_ddot
            omega += df * omega_dot
            theta_dot += df * theta_ddot
            theta += df * theta_dot

            # Handle bar limits (80 deg.)
            theta = jnp.clip(theta, self._state_range[3, 0], self._state_range[3, 1])

            # Update position (x,y) of tires
            front_term = psi + theta + jnp.sign(psi + theta) * jnp.arcsin(self._v * df / (2.0 * r_f))
            back_term = psi + jnp.sign(psi) * jnp.arcsin(self._v * df / (2.0 * r_b))
            x_f += -jnp.sin(front_term)
            y_f += jnp.cos(front_term)
            x_b += -jnp.sin(back_term)
            y_b += jnp.cos(back_term)

            # Handle Roundoff errors, to keep the length of the bicycle
            # constant
            dist = jnp.sqrt((x_f - x_b) ** 2 + (y_f - y_b) ** 2)
            if jnp.abs(dist - self._l) > 0.01:
                x_b += (x_b - x_f) * (self._l - dist) / dist
                y_b += (y_b - y_f) * (self._l - dist) / dist

            # Update psi
            if x_f == x_b and y_f - y_b < 0:
                psi = jnp.pi
            elif y_f - y_b > 0:
                psi = jnp.arctan((x_b - x_f) / (y_f - y_b))
            else:
                psi = jnp.sign(x_b - x_f) * (jnp.pi / 2.0) - jnp.arctan((y_f - y_b) / (x_b - x_f))

        self._state = jnp.array([omega, omega_dot, omega_ddot, theta, theta_dot])
        self._position = jnp.array([x_f, y_f, x_b, y_b, psi])

        reward = 0
        if jnp.abs(omega) > self._state_range[0, 1]:  # Bicycle fell over
            self._absorbing = True
            reward = -1.0
        elif self._isAtGoal():
            self._absorbing = True
            reward = self._reward_goal
        elif not self._navigate:
            self._absorbing = False
            reward = self._reward_shaping
        else:
            goal_angle = self._angle_between(self._goal_loc, jnp.array([x_f - x_b, y_f - y_b])) * jnp.pi / 180.0

            self._absorbing = False
            # return (4. - goal_angle**2) * self.reward_shaping
            # ret =  0.1 * (self.angleWrapPi(old_psi) - self.angleWrapPi(psi))
            ret = 0.1 * (self._angleWrapPi(goal_angle_old) - self._angleWrapPi(goal_angle))
            reward = ret
        return self._getState(), reward, self._absorbing, {}

    def _unit_vector(self, vector):
        """Returns the unit vector of the vector."""
        return vector / jnp.linalg.norm(vector)

    def _angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'::
        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
        """
        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)
        return jnp.arccos(jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0))

    def _isAtGoal(self):
        # Anywhere in the goal radius
        if self._navigate:
            return jnp.sqrt(max(0.0, ((self._position[:2] - self._goal_loc) ** 2).sum() - self._goal_rsqrd)) < 1.0e-5
        else:
            return False

    def _getState(self):
        omega, omega_dot, omega_ddot, theta, theta_dot = tuple(self._state)
        x_f, y_f, x_b, y_b, psi = tuple(self._position)
        goal_angle = self._angle_between(self._goal_loc, jnp.array([x_f - x_b, y_f - y_b])) * jnp.pi / 180.0
        """ modified to follow Ernst paper"""
        return jnp.array([omega, omega_dot, theta, theta_dot, goal_angle])

    def _angleWrapPi(self, x):
        while x < -jnp.pi:
            x += 2.0 * jnp.pi
        while x > jnp.pi:
            x -= 2.0 * jnp.pi
        return x
