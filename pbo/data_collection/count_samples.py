import numpy as np


from pbo.data_collection.replay_buffer import ReplayBuffer


def count_samples(
    replay_buffer: ReplayBuffer, max_state: float, n_discrete_states: int, max_action: float, n_discrete_actions: int
) -> tuple:
    # for each state, get the index where it is located in the discrete states.
    state_box_half_size = max_state / n_discrete_states
    discrete_states_boxes = np.linspace(
        -max_state - state_box_half_size, max_state + state_box_half_size, n_discrete_states + 1
    )
    states = np.array(replay_buffer.states).reshape(-1)
    indexes_states_boxes = np.searchsorted(discrete_states_boxes, states) - 1

    # for each actions, get the index where it is located in the discrete actions.
    action_box_half_size = max_action / n_discrete_actions
    discrete_actions_boxes = np.linspace(
        -max_action - action_box_half_size, max_action + action_box_half_size, n_discrete_actions + 1
    )
    actions = np.array(replay_buffer.actions).reshape(-1)
    indexes_actions_boxes = np.searchsorted(discrete_actions_boxes, actions) - 1

    # only count the state action pairs that are in the boxes
    states_inside_boxes = np.logical_and(
        states >= -max_state - state_box_half_size, states <= max_state + state_box_half_size
    )
    actions_inside_boxes = np.logical_and(
        actions >= -max_action - action_box_half_size, actions <= max_action + action_box_half_size
    )
    states_actions_inside_boxes = np.logical_and(states_inside_boxes, actions_inside_boxes)

    samples_count = np.zeros((n_discrete_states, n_discrete_actions))
    for idx_state, idx_action in zip(
        indexes_states_boxes[states_actions_inside_boxes], indexes_actions_boxes[states_actions_inside_boxes]
    ):
        samples_count[idx_state, idx_action] += 1

    return samples_count, (~states_actions_inside_boxes).sum()
