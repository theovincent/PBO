import numpy as np


from pbo.sample_collection.replay_buffer import ReplayBuffer


def count_samples(
    replay_buffer: ReplayBuffer, discrete_states_boxes: np.ndarray, discrete_actions_boxes: np.ndarray
) -> tuple:
    # for each state, get the index where it is located in the discrete states.
    states = np.array(replay_buffer.states).reshape(-1)
    indexes_states_boxes = np.searchsorted(discrete_states_boxes, states) - 1

    # for each actions, get the index where it is located in the discrete actions.
    actions = np.array(replay_buffer.actions).reshape(-1)
    indexes_actions_boxes = np.searchsorted(discrete_actions_boxes, actions) - 1

    # only count the state action pairs that are in the boxes
    states_inside_boxes = np.logical_and(states >= discrete_states_boxes[0], states <= discrete_states_boxes[-1])
    actions_inside_boxes = np.logical_and(actions >= discrete_actions_boxes[0], actions <= discrete_actions_boxes[-1])
    states_actions_inside_boxes = np.logical_and(states_inside_boxes, actions_inside_boxes)

    samples_count = np.zeros((len(discrete_states_boxes) - 1, len(discrete_actions_boxes) - 1))
    for idx_state, idx_action in zip(
        indexes_states_boxes[states_actions_inside_boxes], indexes_actions_boxes[states_actions_inside_boxes]
    ):
        samples_count[idx_state, idx_action] += 1

    return samples_count, (~states_actions_inside_boxes).sum()
