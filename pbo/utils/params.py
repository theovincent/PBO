from typing import Tuple
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


def load_pickled_data(path: str):
    with open(path, "rb") as handle:
        object = pickle.load(handle)

    return object


class ParameterConverter:
    def __init__(self, params: FrozenDict) -> None:
        self.weights_dimension = sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
        self.weights_information = []
        current_index = 0

        for path, leaf in jax.tree_util.tree_leaves_with_path(params):
            self.weights_information.append(
                (
                    path,
                    {
                        "begin_idx": current_index,
                        "end_idx": current_index + int(np.prod(leaf.shape)),
                        "shape": leaf.shape,
                    },
                )
            )
            current_index += int(np.prod(leaf.shape))

        self.params = jax.tree_util.tree_map(lambda x: x * 0, params)

    @staticmethod
    def get_leaf(dict_: FrozenDict, path: Tuple):
        return dict_[path[0].key][path[1].key][path[2].key]

    @staticmethod
    def set_leaf(dict_: FrozenDict, path: Tuple, value: jnp.ndarray):
        dict_[path[0].key][path[1].key][path[2].key] = value

    @partial(jax.jit, static_argnames="self")
    def to_params(self, weights: jnp.ndarray) -> FrozenDict:
        params = self.params.copy()

        for path, info in self.weights_information:
            self.set_leaf(params, path, weights[info["begin_idx"] : info["end_idx"]].reshape(info["shape"]))

        return params

    @partial(jax.jit, static_argnames="self")
    def to_weights(self, params: FrozenDict) -> jnp.ndarray:
        weights = jnp.zeros(self.weights_dimension)

        for path, info in self.weights_information:
            weights = weights.at[info["begin_idx"] : info["end_idx"]].set(jnp.ravel(self.get_leaf(params, path)))

        return weights
