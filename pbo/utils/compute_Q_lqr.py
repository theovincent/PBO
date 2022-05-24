import numpy as np

from pbo.environment.linear_quadratic import LinearQuadraticEnv


def get_Q_value(env: LinearQuadraticEnv, state: np.ndarray, action: np.ndarray, gamma: float) -> float:
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

        discount_factor *= gamma
        q_value += discount_factor * reward[0]
        step += 1

    assert (
        abs(reward[0]) < 1e-9
    ), f"Last reward: {reward[0]} was not zero. It is most likely coming from the fact that the action are cropped before getting in the environment."

    return q_value


def compute_Q_lqr(
    env: LinearQuadraticEnv,
    states: np.ndarray,
    actions: np.ndarray,
    gamma: float,
) -> np.ndarray:
    Q_values = np.zeros((len(states), len(actions)))

    for idx_state, state in enumerate(states):
        for idx_action, action in enumerate(actions):
            Q_values[idx_state, idx_action] = get_Q_value(env, np.array([state]), np.array([action]), gamma)

    return Q_values
