from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk

from pbo.networks.base_q import BaseQ


class ChainWalkEnv:
    def __init__(
        self,
        env_key: jax.random.PRNGKeyArray,
        n_states: int,
        sucess_probability: float,
        gamma: float,
    ) -> None:
        self.n_states = n_states
        self.states = jnp.arange(self.n_states)
        self.n_actions = 2
        self.actions = jnp.arange(self.n_actions)
        states_mesh, actions_mesh = jnp.meshgrid(self.states, self.actions, indexing="ij")
        self.states_grid = states_mesh.reshape((-1, 1))
        self.actions_grid = actions_mesh.reshape((-1, 1))

        self.sucess_probability = sucess_probability
        self.gamma = gamma
        self.noise_key = env_key

        self.rewards = jnp.zeros(self.n_states)
        self.rewards = self.rewards.at[0].set(1)
        self.rewards = self.rewards.at[-1].set(1)

        self.transition_proba = jnp.zeros((n_states * self.n_actions, n_states))

        for state in range(n_states):
            for action in range(self.n_actions):
                if state != 0 and state != n_states - 1:
                    self.transition_proba = self.transition_proba.at[state * self.n_actions + action, state].set(
                        1 - sucess_probability
                    )
                    self.transition_proba = self.transition_proba.at[
                        state * self.n_actions + action, state + 2 * action - 1
                    ].set(sucess_probability)
                else:
                    self.transition_proba = self.transition_proba.at[state * self.n_actions + action, state].set(1)

        self.R = jnp.zeros(n_states * self.n_actions)
        self.R = self.R.at[0].set(1)
        self.R = self.R.at[1].set(1)
        self.R = self.R.at[-1].set(1)
        self.R = self.R.at[-2].set(1)

    def reset(self, state: jnp.ndarray) -> jnp.ndarray:
        self.state = state

        return self.state

    def step(self, action: jnp.ndarray) -> tuple:
        if 0 < self.state[0] < self.n_states - 1:
            self.noise_key, key = jax.random.split(self.noise_key)
            if jax.random.uniform(key) <= self.sucess_probability:
                self.state = self.state - 1 if action[0] == 0 else self.state + 1
            reward = 0
        else:
            reward = 1

        return self.state, jnp.array([reward]), jnp.array([False]), {}

    @partial(jax.jit, static_argnames="self")
    def apply_bellman_operator(self, q: jnp.ndarray) -> jnp.array:
        return (self.R + self.gamma * self.transition_proba @ jnp.max(q, axis=1)).reshape(
            (self.n_states, self.n_actions)
        )

    def optimal_Q_mesh(self) -> jnp.ndarray:
        q = jnp.zeros((self.n_states, 2))
        changes = float("inf")

        while changes > 1e-9:
            old_q = q.copy()
            q = self.apply_bellman_operator(q)
            changes = jnp.linalg.norm(q - old_q)

        return jnp.array(q)

    @partial(jax.jit, static_argnames="self")
    def policy_transition_probability(self, policy: jnp.ndarray) -> jnp.ndarray:
        policy_transition_proba = jnp.zeros((self.n_states, self.n_states))

        for state in jnp.arange(self.n_states):
            for next_state in jnp.arange(self.n_states):
                policy_transition_proba = policy_transition_proba.at[state, next_state].set(
                    self.transition_proba[state * self.n_actions + policy[state], next_state]
                )

        return policy_transition_proba

    @partial(jax.jit, static_argnames="self")
    def value_function(self, policy: jnp.ndarray) -> jnp.ndarray:
        policy_transition_probability = self.policy_transition_probability(policy)
        return jnp.linalg.solve(jnp.eye(self.n_states) - self.gamma * policy_transition_probability, self.rewards)

    @partial(jax.jit, static_argnames=("self", "q"))
    def discretize(self, q: BaseQ, q_params: hk.Params) -> jnp.ndarray:
        # Dangerous reshape: the indexing of meshgrid is 'ij'.
        return q(q_params, self.states_grid, self.actions_grid).reshape((self.n_states, self.n_actions))
