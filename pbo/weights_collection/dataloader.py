import jax
import jax.numpy as jnp

from pbo.networks.pbo import BasePBO
from pbo.weights_collection.weights_buffer import WeightsBuffer


class WeightsDataLoader:
    def __init__(self, weights_buffer: WeightsBuffer, batch_size: int, shuffle_key: int) -> None:
        self.weights = weights_buffer.weights
        self.n_weights = len(weights_buffer)
        self.batch_size = batch_size
        self.shuffle_key = shuffle_key

        self.indexes = jnp.arange(0, self.n_weights)

        self.jitted_getitem = jax.jit(self.getitem)

    def __len__(self) -> int:
        return jnp.ceil(self.n_weights / self.batch_size).astype(int)

    def getitem(self, idxs) -> dict:
        return self.weights[idxs]

    def __getitem__(self, idx: int) -> dict:
        assert 0 <= idx and idx <= len(self), f"The queried index {idx} is out of scope [0, {len(self)}]."
        if idx == len(self):
            raise StopIteration

        start = idx * self.batch_size
        end = jnp.minimum((idx + 1) * self.batch_size, self.n_weights)
        idxs = self.indexes[start:end]

        return self.jitted_getitem(idxs)

    def shuffle(self) -> None:
        self.shuffle_key, key = jax.random.split(self.shuffle_key)
        self.indexes = jax.random.permutation(key, self.indexes)
