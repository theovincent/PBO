from functools import partial

import jax
import jax.numpy as jnp


class ChainWalkEnv:
    def __init__(
        self,
        env_key: int,
        n_states: int,
        sucess_probability: float,
        gamma: float,
        initial_state: jnp.ndarray = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = 2
        self.sucess_probability = sucess_probability
        self.gamma = gamma
        self.next_state_key, self.reset_key = jax.random.split(env_key)

        self.initial_state = initial_state

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

        self.PR = self.transition_proba @ self.rewards

    def reset(self, state: jnp.ndarray = None) -> jnp.ndarray:
        if state is None:
            if self.initial_state is not None:
                self.state = self.initial_state
            else:
                self.state = jax.random.randint(self.reset_key, [1], minval=0, maxval=self.n_states)
        else:
            self.state = state

        return self.state

    def step(self, action: jnp.ndarray) -> tuple:
        if 0 < self.state[0] < self.n_states - 1:
            self.next_state_key, key = jax.random.split(self.next_state_key)
            if jax.random.uniform(key) <= self.sucess_probability:
                self.state = self.state - 1 if action[0] == 0 else self.state + 1
            reward = 0
        else:
            reward = 1

        absorbing = False if 0 < self.state[0] < self.n_states - 1 else True

        return self.state, jnp.array([reward]), absorbing, {}

    @partial(jax.jit, static_argnames="self")
    def apply_bellman_operator(self, q: jnp.ndarray) -> jnp.array:
        return (self.PR + self.gamma * self.transition_proba @ jnp.max(q, axis=1)).reshape(
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

        for next_state in jnp.arange(self.n_states):
            for state in jnp.arange(self.n_states):
                policy_transition_proba = policy_transition_proba.at[next_state, state].set(
                    self.transition_proba[state * self.n_actions + policy[state], next_state]
                )

        return policy_transition_proba

    @partial(jax.jit, static_argnames=("self", "horizon"))
    def absorbing_probability(self, policy: jnp.ndarray, horizon: int) -> jnp.ndarray:
        absorbing_proba = jnp.zeros(horizon)
        policy_transition_proba = self.policy_transition_probability(policy)
        states_distribution = jnp.ones(self.n_states) / self.n_states

        for time in jnp.arange(horizon):
            states_distribution = policy_transition_proba @ states_distribution
            absorbing_proba = absorbing_proba.at[time].set(states_distribution[0] + states_distribution[-1])

        return absorbing_proba

    def rollout(self, policy: jnp.ndarray, initial_state: jnp.ndarray, horizon: int) -> float:
        self.state = initial_state
        rollout_return = 0
        discount = 1

        for _ in jnp.arange(horizon):
            _, reward, _, _ = self.step(policy[self.state[0]])

            rollout_return += discount * reward
            discount *= self.gamma

        return rollout_return

    def monte_carlo(self, policy: jnp.ndarray, initial_state: jnp.ndarray, horizon: int, n_rollout: int) -> float:
        monte_carlo_estimate = 0

        for _ in jnp.arange(n_rollout):
            monte_carlo_estimate += self.rollout(policy, initial_state, horizon)

        return monte_carlo_estimate / n_rollout
