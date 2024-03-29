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

    def __init__(self, env_key: jax.random.PRNGKeyArray, gamma: float) -> None:
        """
        state = [omega, omega_dot, theta, theta_dot]
        position = [x_b, y_b, x_f, y_f, psi]
        """
        self.noise_key = env_key
        self.gamma = gamma
        self.actions_on_max = jnp.array(
            [[-1, 0], [0, 0], [1, 0], [0, -1], [0, 1]]
        )  # , [-1, -1], [-1, 1], [1, -1], [1, 1]])
        self.idx_actions_with_d_1 = jnp.nonzero(self.actions_on_max[:, 0] == 1)[0].flatten()
        self.idx_actions_with_d_m1 = jnp.nonzero(self.actions_on_max[:, 0] == -1)[0].flatten()
        self.idx_actions_with_T_1 = jnp.nonzero(self.actions_on_max[:, 1] == 1)[0].flatten()
        self.idx_actions_with_T_m1 = jnp.nonzero(self.actions_on_max[:, 1] == -1)[0].flatten()

        self.noise = 0.02 / 10
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
        self.position = self.position.at[2].set(self._l * jnp.cos(self.state[4]))
        self.position = self.position.at[3].set(self._l * jnp.sin(self.state[4]))
        self.positions = [self.position]
        self.max_distance = self._l

        self.n_steps = 0

        return self.state

    @partial(jax.jit, static_argnames=("self"))
    def jitted_step(self, action, noise_key, state, position):
        # action in [-1, 0, 1] x [-1, 0, 1]
        d = 0.02 * action[0]  # Displacement of center of mass (in meters)
        T = 2.0 * action[1]  # Torque on handle bars

        # Add noise to action
        d += jax.random.uniform(noise_key, minval=-1, maxval=1) * self.noise

        omega_t, omega_dot_t, theta_t, theta_dot_t = state
        x_b_t, y_b_t, _, _, psi_t = position

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

        # Update psi
        psi_t1 = psi_t + self._v * self._dt * jnp.sign(theta_t) * inv_r_b
        psi_t1 = jnp.angle(jnp.exp(psi_t1 * 1j))

        # Update positions
        x_b_t1 = x_b_t + self._v * self._dt * jnp.cos(psi_t)
        y_b_t1 = y_b_t + self._v * self._dt * jnp.sin(psi_t)
        x_f_t1 = x_b_t1 + self._l * jnp.cos(psi_t)
        y_f_t1 = y_b_t1 + self._l * jnp.sin(psi_t)

        next_state = jnp.array([omega_t1, omega_dot_t1, theta_t1, theta_dot_t1])
        next_position = jnp.array([x_b_t1, y_b_t1, x_f_t1, y_f_t1, psi_t1])

        # Reward and absorbing
        reward = -1 * (jnp.abs(omega_t1) > self.omega_bound) + 10000 * (jnp.abs(omega_t) - jnp.abs(omega_t1))
        absorbing = (jnp.abs(omega_t1) > self.omega_bound) * jnp.array([1])

        return next_state, next_position, jnp.array([reward]), absorbing.astype(bool)

    def step(self, action: jnp.ndarray) -> jnp.ndarray:
        self.noise_key, key = jax.random.split(self.noise_key)
        self.state, self.position, reward, absorbing = self.jitted_step(action, key, self.state, self.position)

        self.n_steps += 1

        return self.state, reward, absorbing, {}

    @partial(jax.jit, static_argnames=("self", "q"))
    def jitted_best_action(self, q: BaseQ, q_params: hk.Params, state: jnp.ndarray) -> jnp.ndarray:
        state_repeat = jnp.repeat(state.reshape((1, 4)), self.actions_on_max.shape[0], axis=0)

        return self.actions_on_max[q(q_params, state_repeat, self.actions_on_max).argmax()]

    @partial(jax.jit, static_argnames=("self", "q"))
    def jitted_best_action_multi_head(self, q: BaseQ, q_params: hk.Params, state: jnp.ndarray) -> jnp.ndarray:
        state_repeat = jnp.repeat(state.reshape((1, 4)), self.actions_on_max.shape[0], axis=0)

        return self.actions_on_max[q(q_params, state_repeat, self.actions_on_max)[:, -1].argmax()]

    def evaluate(self, q: BaseQ, q_params: hk.Params, horizon: int, n_simulations: int) -> np.ndarray:
        rewards = np.zeros((n_simulations, 2))

        for idx_simulation in range(n_simulations):
            self.reset()
            absorbing = False
            cumulative_reward = 0
            discount = 1

            while not absorbing and self.n_steps < horizon:
                best_action = self.jitted_best_action(q, q_params, self.state)
                _, reward, absorbing, _ = self.step(best_action)

                cumulative_reward += discount * reward
                discount *= self.gamma

            rewards[idx_simulation] = np.array([self.n_steps, cumulative_reward[0]])

        return rewards.mean(axis=0)

    def collect_positions(self, q: BaseQ, q_params: hk.Params, horizon: int) -> jnp.ndarray:
        self.reset()
        absorbing = False

        while not absorbing and self.n_steps < horizon:
            best_action = self.jitted_best_action(q, q_params, self.state)
            _, _, absorbing, _ = self.step(best_action)

            self.positions.append(self.position)

        return jnp.array(self.positions)[:, :2]

    @partial(jax.jit, static_argnames=("self", "q"))
    def best_action_on_omegas(
        self,
        q: BaseQ,
        q_params: hk.Params,
        omegas: jnp.ndarray,
        omega_dots: jnp.ndarray,
        sample_thetas_theta_dots: jnp.ndarray,
    ) -> jnp.ndarray:
        n_omegas = omegas.shape[0]
        n_omega_dots = omega_dots.shape[0]
        n_sample_thetas = sample_thetas_theta_dots.shape[0]
        n_boxes = n_omegas * n_omega_dots * n_sample_thetas * self.actions_on_max.shape[0]

        idx_omegas_mesh, idx_omega_dots_mesh, idx_thetas_theta_dots_mesh, idx_actions_mesh = jnp.meshgrid(
            jnp.arange(n_omegas),
            jnp.arange(n_omega_dots),
            jnp.arange(n_sample_thetas),
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
            (n_omegas * n_omega_dots * n_sample_thetas, self.actions_on_max.shape[0])
        )

        best_actions = self.actions_on_max[q_values.argmax(axis=1), 0]

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        return best_actions.reshape((n_omegas, n_omega_dots, n_sample_thetas)).mean(axis=2)

    @partial(jax.jit, static_argnames=("self", "q"))
    def best_action_on_thetas(
        self,
        q: BaseQ,
        q_params: hk.Params,
        samples_omegas_omegas_dots: jnp.ndarray,
        thetas: jnp.ndarray,
        theta_dots: jnp.ndarray,
    ) -> jnp.ndarray:
        n_sample_omegas = samples_omegas_omegas_dots.shape[0]
        n_thetas = thetas.shape[0]
        n_theta_dots = theta_dots.shape[0]
        n_boxes = n_sample_omegas * n_thetas * n_theta_dots * self.actions_on_max.shape[0]

        (idx_omegas_omega_dots_mesh, idx_thetas_mesh, idx_theta_dots_mesh, idx_actions_mesh,) = jnp.meshgrid(
            jnp.arange(n_sample_omegas),
            jnp.arange(n_thetas),
            jnp.arange(n_theta_dots),
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
            (n_sample_omegas * n_thetas * n_theta_dots, self.actions_on_max.shape[0])
        )

        best_actions = self.actions_on_max[q_values.argmax(axis=1), 1]

        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        return best_actions.reshape((n_sample_omegas, n_thetas, n_theta_dots)).mean(axis=0)

    def render(self, action: jnp.ndarray = None) -> None:
        # Store position
        self.positions.append(self.position[:4])

        # Update max distance
        distance = jnp.maximum(
            jnp.linalg.norm(self.positions[-1][jnp.array([0, 1])]),
            jnp.linalg.norm(self.positions[-1][jnp.array([2, 3])]),
        )
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
                self._viewer._translate(position[jnp.array([0, 1])] / (2.1 * self.max_distance), center_top),
                self._viewer._translate(position[jnp.array([2, 3])] / (2.1 * self.max_distance), center_top),
            )

        # Display max distance
        self._viewer.text([0, 2], f"Max distance: {jnp.round(self.max_distance, 1)}")

        self._viewer.display(self._dt)

    def close(self):
        return self._viewer.close()
