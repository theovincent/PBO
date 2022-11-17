import pickle
import jax
import haiku as hk


def save_params(path: str, params: hk.Params) -> None:
    params = jax.device_get(params)
    with open(path, "wb") as fp:
        pickle.dump(params, fp)


def load_params(path: str) -> hk.Params:
    with open(path, "rb") as fp:
        params = pickle.load(fp)
    return jax.device_put(params)
