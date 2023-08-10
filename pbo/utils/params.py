from functools import partial
from flax.core import FrozenDict
import numpy as np
import jax
import jax.numpy as jnp
import pickle


def save_pickled_data(path: str, object):
    object = jax.device_get(object)

    with open(path, "wb") as handle:
        pickle.dump(object, handle)


def load_pickled_data(path: str, device_put: bool = False):
    with open(path, "rb") as handle:
        object = pickle.load(handle)

    if device_put:
        return jax.device_put(object)
    else:
        return object


class ParameterConverter:
    def __init__(self, params: FrozenDict) -> None:
        self.weights_information = {}
        self.weights_dimension = 0

        for key_layer, layer in params.items():
            self.weights_information[key_layer] = dict()
            for key_weight_layer, weight_layer in layer.items():
                # int because weight_layer.shape = () can happen
                weight_layer_dimensions = int(np.prod(weight_layer.shape))

                self.weights_information[key_layer][key_weight_layer] = {
                    "begin_idx": self.weights_dimension,
                    "end_idx": self.weights_dimension + weight_layer_dimensions,
                    "shape": weight_layer.shape,
                }
                self.weights_dimension += weight_layer_dimensions

    @partial(jax.jit, static_argnames="self")
    def to_params(self, weights: jnp.ndarray) -> FrozenDict:
        params = dict()

        for key_layer, layer_info in self.weights_information.items():
            params[key_layer] = dict()
            for key_weight_layer, weight_layer_info in layer_info.items():
                begin_idx = weight_layer_info["begin_idx"]
                end_idx = weight_layer_info["end_idx"]
                shape = weight_layer_info["shape"]

                params[key_layer][key_weight_layer] = weights[begin_idx:end_idx].reshape(shape)

        return params

    @partial(jax.jit, static_argnames="self")
    def to_weights(self, params: FrozenDict) -> jnp.ndarray:
        weights = jnp.zeros(self.weights_dimension)

        for key_layer, layer in params.items():
            for key_weight_layer, weight_layer in layer.items():
                begin_idx = self.weights_information[key_layer][key_weight_layer]["begin_idx"]
                end_idx = self.weights_information[key_layer][key_weight_layer]["end_idx"]

                weights = weights.at[begin_idx:end_idx].set(weight_layer.flatten())

        return jnp.array(weights)
