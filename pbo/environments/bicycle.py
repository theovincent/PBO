# This file was inspired by https://github.com/teopir/ifqi

from functools import partial
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp

from pbo.environments.viewer import Viewer
from pbo.networks.base_q import BaseQ


class BicycleEnv:
    """
    The Bicycle balancing/riding selfironment as presented in:
    Learning to Drive a Bicycle using Reinforcement Learning and Shaping.
    Jette Randlov and Preben Alstrom. 1998.
    """

    def __init__(self, env_key: int) -> None:
        """
        state = [omega, omega_dot, theta, theta_dot, psi]
        position = [x_b, y_b, x_f, y_f]
        """
        self.noise_key = env_key
        self.actions_on_max = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])
        self.idx_actions_with_d_1 = jnp.nonzero(self.actions_on_max[:, 0] == 1)[0].flatten()
        self.idx_actions_with_d_m1 = jnp.nonzero(self.actions_on_max[:, 0] == -1)[0].flatten()
        self.idx_actions_with_T_1 = jnp.nonzero(self.actions_on_max[:, 1] == 1)[0].flatten()
        self.idx_actions_with_T_m1 = jnp.nonzero(self.actions_on_max[:, 1] == -1)[0].flatten()

        self.noise = 0.02
        self.omega_bound = jnp.pi * 12.0 / 180.0
        self.theta_bound = jnp.pi * 80.0 / 180.0

        # Units in Meters and Kilograms
        self._c = 0.66  # Horizontal distance between bottom of front wheel and center of mass
        self._d_cm = 0.30  # Vertical dist between center of mass and the cyclist
        self._h = 0.94  # Height of the center of mass over the ground
        self._l = 1.11  # Distance between front tire and back tire at point on ground
        self._M_c = 15.0  # Mass of bicycle
        self._M_d = 1.7  # Mass of tire
        self._M_p = 60.0  # Mass of cyclist
        self._r = 0.34  # Radius of tire
        self._v = 10.0 / 3.6  # Velocity of bicycle (converted from km/h to m/s)

        # Useful precomputations
        self._M = self._M_p + self._M_c
        self._Inertia_bc = (13.0 / 3.0) * self._M_c * self._h**2 + self._M_p * (self._h + self._d_cm) ** 2
        self._Inertia_dc = self._M_d * self._r**2
        self._Inertia_dv = 3 / 2 * self._M_d * self._r**2
        self._Inertia_dl = 1 / 2 * self._M_d * self._r**2
        self._sigma_dot = self._v / self._r

        # Simulation Constants
        self._g = 9.82
        self._dt = 0.01

        # Visualization
        self._viewer = Viewer(2, 2, width=1000, height=1000)

    def reset(self, state: jnp.ndarray = None) -> jnp.ndarray:
        if state is None:
            self.state = jnp.zeros((4))
        else:
            self.state = state

        self.position = jnp.zeros((5))
        self.position = self.position.at[2].set(self._l * jnp.cos(self.state[-1]))
        self.position = self.position.at[3].set(self._l * jnp.sin(self.state[-1]))
        self.positions = [self.position]
        self.max_distance = self._l

        return self.state

    @partial(jax.jit, static_argnames="self")
    def step_jitted(self, action, noise_key, state, position):
        # action in [-1, 0, 1] x [-1, 0, 1]
        d = 0.02 * action[0]  # Displacement of center of mass (in meters)
        T = 2.0 * action[1]  # Torque on handle bars

        # Add noise to action
        d += jax.random.uniform(noise_key, minval=-1, maxval=1) * self.noise

        omega_t, omega_dot_t, theta_t, theta_dot_t = state

        phi_t = omega_t + jnp.arctan(d / self._h)

        inv_r_f = jnp.abs(jnp.sin(theta_t)) / self._l
        inv_r_b = jnp.abs(jnp.tan(theta_t)) / self._l
        inv_r_CM = 1 / jnp.sqrt((self._l - self._c) ** 2 + (self._l**2 / (jnp.tan(theta_t) ** 2 + 1e-32)))

        # Update omega
        omega_ddot_t = self._M * self._h * self._g * jnp.sin(phi_t) - jnp.cos(phi_t) * (
            self._Inertia_dc * self._sigma_dot * theta_dot_t
            + jnp.sign(theta_t)
            * self._v**2
            * (self._M_d * self._r * inv_r_f + self._M_d * self._r * inv_r_b + self._M * self._h * inv_r_CM)
        )
        omega_ddot_t /= self._Inertia_bc
        omega_t1 = omega_t + self._dt * omega_dot_t
        omega_dot_t1 = omega_dot_t + self._dt * omega_ddot_t

        # Update theta
        theta_ddot_t = (T - self._Inertia_dv * self._sigma_dot * omega_dot_t) / self._Inertia_dl
        theta_t1 = theta_t + self._dt * theta_dot_t
        theta_dot_t1 = (jnp.abs(theta_t1) <= self.theta_bound) * (theta_dot_t + self._dt * theta_ddot_t)
        theta_t1 = jnp.clip(theta_t1, -self.theta_bound, self.theta_bound)  # Handle bar angle limits

        # Update positions
        x_b_t, y_b_t, _, _, psi_t = position

        x_b_t1 = x_b_t + self._v * self._dt * jnp.cos(psi_t)
        y_b_t1 = y_b_t + self._v * self._dt * jnp.sin(psi_t)
        x_f_t1 = x_b_t1 + self._l * jnp.cos(psi_t)
        y_f_t1 = y_b_t1 + self._l * jnp.sin(psi_t)

        psi_t1 = psi_t + self._v * self._dt * jnp.sign(theta_t) * inv_r_b

        next_state = jnp.array([omega_t1, omega_dot_t1, theta_t1, theta_dot_t1])
        next_position = jnp.array([x_b_t1, y_b_t1, x_f_t1, y_f_t1, psi_t1])

        # Reward and absorbing
        reward = (jnp.abs(omega_t1) > self.omega_bound) * jnp.array([-1]) + 100 * (jnp.abs(omega_t) - jnp.abs(omega_t1))
        absorbing = (jnp.abs(omega_t1) > self.omega_bound) * jnp.array([1])

        return next_state, next_position, reward, absorbing.astype(bool)

    def step(self, action: jnp.ndarray) -> jnp.ndarray:
        self.noise_key, key = jax.random.split(self.noise_key)
        self.state, self.position, reward, absorbing = self.step_jitted(action, key, self.state, self.position)

        return self.state, reward, absorbing, {}

    def render(self, action: jnp.ndarray = None) -> None:
        # Store position
        self.positions.append(self.position)

        # Update max distance
        distance = jnp.maximum(jnp.linalg.norm(self.position[:2]), jnp.linalg.norm(self.position[2:-1]))
        if distance > self.max_distance:
            self.max_distance = distance

        # Plot
        dark_blue = (102, 153, 255)
        light_blue = (131, 247, 228)
        grey = (200, 200, 200)
        red = (255, 0, 0)

        omega, _, theta, _ = self.state
        _, _, _, _, psi = self.position

        # Split in three screens
        self._viewer.line([1, 0], [1, 1], width=2)
        self._viewer.line([0, 1], [2, 1], width=2)
        center_left = [0.5, 0.5]
        center_right = [1.5, 0.1]
        center_top = [1, 1.5]

        # --- Left plot --- #
        # Axes
        self._viewer.text(self._viewer._translate([0.4, -0.04], center_left), "x")
        self._viewer.line(
            self._viewer._translate([-0.4, 0], center_left), self._viewer._translate([0.4, 0], center_left), width=2
        )
        self._viewer.arrow_head(self._viewer._translate([0.4, 0], center_left), 0.05, 0)
        self._viewer.text(self._viewer._translate([-0.06, 0.4], center_left), "y")
        self._viewer.line(
            self._viewer._translate([0, -0.4], center_left), self._viewer._translate([0, 0.4], center_left), width=2
        )
        self._viewer.arrow_head(self._viewer._translate([0, 0.4], center_left), 0.05, np.pi / 2)
        self._viewer.text(self._viewer._translate([-0.45, -0.4], center_left), "z")
        self._viewer.circle(self._viewer._translate([-0.4, -0.4], center_left), 0.01, width=1)
        self._viewer.circle(self._viewer._translate([-0.4, -0.4], center_left), 0.005, width=2)

        # Bicycle
        self._viewer.line(
            self._viewer._translate(self._viewer._rotate([-0.3, 0], psi), center_left),
            self._viewer._translate(self._viewer._rotate([0.3, 0], psi), center_left),
            color=dark_blue,
            width=5,
        )

        # Handbar
        self._viewer.line(
            self._viewer._translate(
                self._viewer._rotate([0, -0.15], psi), center_left + self._viewer._rotate([0.3, 0], psi)
            ),
            self._viewer._translate(
                self._viewer._rotate([0, 0.15], psi), center_left + self._viewer._rotate([0.3, 0], psi)
            ),
            color=grey,
            width=1,
        )
        self._viewer.line(
            self._viewer._translate(
                self._viewer._rotate([0, -0.15], psi + theta), center_left + self._viewer._rotate([0.3, 0], psi)
            ),
            self._viewer._translate(
                self._viewer._rotate([0, 0.15], psi + theta), center_left + self._viewer._rotate([0.3, 0], psi)
            ),
            color=light_blue,
            width=5,
        )

        # Torque
        if action is not None:
            self._viewer.torque_arrow(center_left + self._viewer._rotate([0.3, 0], psi), action[1], 2, 0.1)

        # --- Right plot --- #
        # Axes
        self._viewer.text(self._viewer._translate([0.4, -0.04], center_right), "x")
        self._viewer.line(
            self._viewer._translate([-0.4, 0], center_right), self._viewer._translate([0.4, 0], center_right), width=2
        )
        self._viewer.arrow_head(self._viewer._translate([0.4, 0], center_right), 0.05, 0)
        self._viewer.text(self._viewer._translate([-0.06, 0.7], center_right), "z")
        self._viewer.line(
            self._viewer._translate([0, -0.1], center_right), self._viewer._translate([0, 0.7], center_right), width=2
        )
        self._viewer.arrow_head(self._viewer._translate([0, 0.7], center_right), 0.05, np.pi / 2)
        self._viewer.text(self._viewer._translate([0.4, 0.78], center_right), "y")
        self._viewer.circle(self._viewer._translate([0.4, 0.8], center_right), 0.01, width=1)
        self._viewer.line(
            self._viewer._translate(self._viewer._rotate([-0.01, 0], np.pi / 4), center_right + np.array([0.4, 0.8])),
            self._viewer._translate(self._viewer._rotate([0.01, 0], np.pi / 4), center_right + np.array([0.4, 0.8])),
            width=1,
        )
        self._viewer.line(
            self._viewer._translate(self._viewer._rotate([0, -0.01], np.pi / 4), center_right + np.array([0.4, 0.8])),
            self._viewer._translate(self._viewer._rotate([0, 0.01], np.pi / 4), center_right + np.array([0.4, 0.8])),
            width=1,
        )

        # Bicycle
        self._viewer.line(
            self._viewer._translate([0, 0], center_right),
            self._viewer._translate(self._viewer._rotate([0, 0.7], -self.omega_bound), center_right),
            color=grey,
            width=1,
        )
        self._viewer.line(
            self._viewer._translate([0, 0], center_right),
            self._viewer._translate(self._viewer._rotate([0, 0.7], self.omega_bound), center_right),
            color=grey,
            width=1,
        )
        self._viewer.line(
            self._viewer._translate([0, 0], center_right),
            self._viewer._translate(self._viewer._rotate([0, 0.7], -omega), center_right),
            color=dark_blue,
            width=5,
        )

        # Center of mass
        if action is not None:
            self._viewer.circle(
                self._viewer._translate(
                    self._viewer._rotate([action[0] * 0.02 * 0.7 / self._h, 0.7], -omega), center_right
                ),
                0.01,
                color=red,
                width=10,
            )

        # --- Top plot --- #
        # Axes
        self._viewer.text(self._viewer._translate([0.8, -0.04], center_top), "x")
        self._viewer.line(
            self._viewer._translate([-0.8, 0], center_top), self._viewer._translate([0.8, 0], center_top), width=2
        )
        self._viewer.arrow_head(self._viewer._translate([0.8, 0], center_top), 0.05, 0)
        self._viewer.text(self._viewer._translate([-0.06, 0.4], center_top), "y")
        self._viewer.line(
            self._viewer._translate([0, -0.4], center_top), self._viewer._translate([0, 0.4], center_top), width=2
        )
        self._viewer.arrow_head(self._viewer._translate([0, 0.4], center_top), 0.05, np.pi / 2)
        self._viewer.text(self._viewer._translate([-0.45, -0.4], center_top), "z")
        self._viewer.circle(self._viewer._translate([-0.4, -0.4], center_top), 0.01, width=1)
        self._viewer.circle(self._viewer._translate([-0.4, -0.4], center_top), 0.005, width=2)

        # Add positions
        for position in self.positions:
            self._viewer.line(
                self._viewer._translate(position[:2] / (2.1 * self.max_distance), center_top),
                self._viewer._translate(position[2:-1] / (2.1 * self.max_distance), center_top),
            )

        # Display max distance
        self._viewer.text([0, 2], f"Max distance: {jnp.round(self.max_distance, 1)}")

        self._viewer.display(self._dt)

    def close(self):
        return self._viewer.close()

    def best_d(self) -> float:
        if (self.state[0] > 0 and self.state[1] >= 0) or (self.state[0] > self.omega_bound / 3):
            return -1
        elif (self.state[0] < 0 and self.state[1] <= 0) or (self.state[0] < -self.omega_bound / 3):
            return 1
        else:
            return 0

    def best_T(self) -> float:
        if (self.state[2] > 0 and self.state[3] >= 0) or (self.state[2] > self.theta_bound / 3):
            return -1
        elif (self.state[2] < 0 and self.state[3] <= 0) or (self.state[2] < -self.theta_bound / 3):
            return 1
        else:
            return 0

    def simulate(
        self, q: BaseQ, horizon: int, render: bool, hard_coded_d: bool = False, hard_coded_T: bool = False
    ) -> bool:
        self.reset()
        absorbing = False
        step = 0

        while not absorbing and step < horizon:
            state_repeat = jnp.repeat(self.state.reshape((1, 4)), self.actions_on_max.shape[0], axis=0)
            best_action = self.actions_on_max[q(q.params, state_repeat, self.actions_on_max).argmax()]

            if hard_coded_d:
                best_action = best_action.at[0].set(self.best_d())
            if hard_coded_T:
                best_action = best_action.at[1].set(self.best_T())

            _, _, absorbing, _ = self.step(best_action)

            step += 1
            if render:
                self.render(best_action)

        self.close()

        return step

    @partial(jax.jit, static_argnames=("self", "q"))
    def q_values_on_omegas(
        self,
        q: BaseQ,
        q_params: hk.Params,
        omegas: jnp.ndarray,
        omega_dots: jnp.ndarray,
        sample_thetas_theta_dots: jnp.ndarray,
    ) -> jnp.ndarray:
        n_boxes = (
            omegas.shape[0] * omega_dots.shape[0] * sample_thetas_theta_dots.shape[0] * self.actions_on_max.shape[0]
        )
        idx_omegas_mesh, idx_omega_dots_mesh, idx_thetas_theta_dots_mesh, idx_actions_mesh = jnp.meshgrid(
            jnp.arange(omegas.shape[0]),
            jnp.arange(omega_dots.shape[0]),
            jnp.arange(sample_thetas_theta_dots.shape[0]),
            jnp.arange(self.actions_on_max.shape[0]),
            indexing="ij",
        )

        states = jnp.hstack(
            (
                omegas[idx_omegas_mesh.flatten()].reshape((n_boxes, 1)),
                omega_dots[idx_omega_dots_mesh.flatten()].reshape((n_boxes, 1)),
                sample_thetas_theta_dots[idx_thetas_theta_dots_mesh.flatten(), 0].reshape((n_boxes, 1)),
                sample_thetas_theta_dots[idx_thetas_theta_dots_mesh.flatten(), 1].reshape((n_boxes, 1)),
            )
        )
        actions = self.actions_on_max[idx_actions_mesh.flatten()]

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        q_values = q(q_params, states, actions).reshape(
            (omegas.shape[0] * omega_dots.shape[0] * sample_thetas_theta_dots.shape[0], self.actions_on_max.shape[0])
        )

        q_diff_T = (q_values[:, self.idx_actions_with_T_1] - q_values[:, self.idx_actions_with_T_m1]).mean(axis=1)

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        return q_diff_T.reshape((omegas.shape[0], omega_dots.shape[0], sample_thetas_theta_dots.shape[0])).mean(axis=2)

    @partial(jax.jit, static_argnames=("self", "q"))
    def q_values_on_thetas(
        self,
        q: BaseQ,
        q_params: hk.Params,
        samples_omegas_omegas_dots: jnp.ndarray,
        thetas: jnp.ndarray,
        theta_dots: jnp.ndarray,
    ) -> jnp.ndarray:
        n_boxes = (
            samples_omegas_omegas_dots.shape[0] * thetas.shape[0] * theta_dots.shape[0] * self.actions_on_max.shape[0]
        )
        idx_omegas_omega_dots_mesh, idx_thetas_mesh, idx_theta_dots_mesh, idx_actions_mesh = jnp.meshgrid(
            jnp.arange(samples_omegas_omegas_dots.shape[0]),
            jnp.arange(thetas.shape[0]),
            jnp.arange(theta_dots.shape[0]),
            jnp.arange(self.actions_on_max.shape[0]),
            indexing="ij",
        )

        states = jnp.hstack(
            (
                samples_omegas_omegas_dots[idx_omegas_omega_dots_mesh.flatten(), 0].reshape((n_boxes, 1)),
                samples_omegas_omegas_dots[idx_omegas_omega_dots_mesh.flatten(), 1].reshape((n_boxes, 1)),
                thetas[idx_thetas_mesh.flatten()].reshape((n_boxes, 1)),
                theta_dots[idx_theta_dots_mesh.flatten()].reshape((n_boxes, 1)),
            )
        )
        actions = self.actions_on_max[idx_actions_mesh.flatten()]

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        q_values = q(q_params, states, actions).reshape(
            (samples_omegas_omegas_dots.shape[0] * thetas.shape[0] * theta_dots.shape[0], self.actions_on_max.shape[0])
        )

        q_diff_T = (q_values[:, self.idx_actions_with_T_1] - q_values[:, self.idx_actions_with_T_m1]).mean(axis=1)

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        return q_diff_T.reshape((samples_omegas_omegas_dots.shape[0], thetas.shape[0], theta_dots.shape[0])).mean(
            axis=0
        )
