import numpy as np


from pbo.data_collection.replay_buffer import ReplayBuffer


def count_samples(
    replay_buffer: ReplayBuffer, max_state: float, n_discrete_states: int, max_action: float, n_discrete_actions: int
) -> np.ndarray:
    state_box_half_size = max_state / n_discrete_states
    discrete_states_boxes = np.linspace(
        -max_state - state_box_half_size, max_state + state_box_half_size, n_discrete_states + 1
    )
    # get the index in the discrete states in which each state is
    indexes_states_boxes = np.searchsorted(discrete_states_boxes, np.array(replay_buffer.states).reshape(-1)) - 1

    action_box_half_size = max_action / n_discrete_actions
    discrete_actions_boxes = np.linspace(
        -max_action - action_box_half_size, max_action + action_box_half_size, n_discrete_actions + 1
    )
    # get the index in the discrete actions in which each action is
    indexes_actions_boxes = np.searchsorted(discrete_actions_boxes, np.array(replay_buffer.actions).reshape(-1)) - 1

    samples_count = np.zeros((n_discrete_states, n_discrete_actions))
    for idx_state, idx_action in zip(indexes_states_boxes, indexes_actions_boxes):
        samples_count[idx_state, idx_action] += 1

    return samples_count
