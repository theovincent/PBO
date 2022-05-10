import numpy as np

from pbo.environment.linear_quadratic import LinearQuadraticEnv


def get_Q_value(env: LinearQuadraticEnv, state: np.ndarray, action: np.ndarray) -> float:
    terminal = False
    step = 0
    reward = 0
    discount_factor = 1
    q_value = 0

    env.reset(state=state)

    while step < 50 and not terminal:
        if step == 0:
            _, reward, terminal, _ = env.step(action)
        else:
            _, reward, terminal, _ = env.step(env.optimal_action())

        discount_factor *= env.gamma
        q_value += discount_factor * reward
        step += 1

    assert abs(reward) < 1e-9, f"Last reward: {reward} was not zero."

    return q_value


def compute_Q_lqr(env: LinearQuadraticEnv, n_discrete_states: int, n_discrete_actions: int) -> np.ndarray:
    states = np.linspace(-env.max_state, env.max_state, n_discrete_states)
    actions = np.linspace(-env.max_action, env.max_action, n_discrete_actions)

    Q_values = np.zeros((len(states), len(actions)))

    for idx_state, state in enumerate(states):
        for idx_action, action in enumerate(actions):
            Q_values[idx_state, idx_action] = get_Q_value(env, np.array([state]), np.array([action]))

    return Q_values
