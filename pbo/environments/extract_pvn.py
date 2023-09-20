import os
import subprocess
import tensorflow as tf
import numpy as np
import jax
import tf2jax


def extract_pvn(game):
    path = f"pbo/environments/pvns/{game}"

    if not os.path.exists(path):
        os.makedirs(path)

        subprocess.run(
            [
                "gsutil",
                "-q",
                "-m",
                "cp",
                "-R",
                f"gs://download-dopamine-rl/pvn/{game}/1/*",
                path,
            ]
        )

    pvn = tf.saved_model.load(path)

    # Convert the encoder to a jitted function
    @tf.function
    def forward(x):
        return pvn(x)

    jax_func, _ = tf2jax.convert(forward, np.zeros((1, 84, 84, 4), dtype=np.float32))

    # remove the first input since the function has no parameters
    # remove the second input since it is always an empty dictionary
    return jax.jit(lambda x: np.squeeze(jax_func({}, x)[0]))
