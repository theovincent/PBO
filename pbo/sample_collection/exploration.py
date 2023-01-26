import jax
import optax


class EpsilonGreedySchedule:
    def __init__(self, starting_eps: float, ending_eps: float, duration_eps: int, key: jax.random.PRNGKeyArray) -> None:
        self.epsilon_schedule = optax.linear_schedule(starting_eps, ending_eps, duration_eps)
        self.exploration_step = 0
        self.exploration_key = key

    def explore(self) -> bool:
        self.exploration_step += 1

        self.exploration_key, key = jax.random.split(self.exploration_key)
        return jax.random.uniform(key) < self.epsilon_schedule(self.exploration_step)
