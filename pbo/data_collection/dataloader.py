import numpy as np

from pbo.data_collection.replay_buffer import ReplayBuffer


class DataLoader:
    def __init__(self, replay_buffer: ReplayBuffer, batch_size: int) -> None:
        self.replay_buffer = replay_buffer
        self.replay_buffer.cast_to_tensor()
        self.batch_size = batch_size

        self.indexes = np.arange(0, len(replay_buffer))

    def __len__(self) -> int:
        return np.ceil(len(self.replay_buffer) / self.batch_size).astype(int)

    def __getitem__(self, idx: int) -> dict:
        assert 0 <= idx and idx <= len(self), f"The queried index {idx} is out of scope [0, {len(self)}]."
        if idx == len(self):
            raise StopIteration

        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.replay_buffer))
        idxs = self.indexes[start:end]

        return {
            "state": self.replay_buffer.state[idxs],
            "action": self.replay_buffer.action[idxs],
            "reward": self.replay_buffer.reward[idxs],
            "next_state": self.replay_buffer.next_state[idxs],
        }

    def shuffle(self) -> None:
        np.random.shuffle(self.indexes)
