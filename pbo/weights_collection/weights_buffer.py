import jax.numpy as jnp


class WeightsBuffer:
    def __init__(self) -> None:
        self.weights = []

    def __len__(self) -> int:
        return len(self.weights)

    def add(self, weights: jnp.ndarray) -> None:
        self.weights.append(weights)

    def cast_to_jax_array(self) -> None:
        self.weights = jnp.array(self.weights)

    def cast_to_list(self) -> None:
        self.weights = list(self.weights)
