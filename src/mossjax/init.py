import numpy as np

import jax
import jax.numpy as jnp

from . import weights as weights_mod


def _is_norm_weight(key: str, shape) -> bool:
    return key.endswith(".weight") and len(shape) == 1 and ("norm" in key)


def init_tree(shapes, rng, list_containers, dtype=jnp.float32):
    keys = sorted(shapes)
    rngs = jax.random.split(rng, max(len(keys), 1))
    flat = {}
    for key, sub in zip(keys, rngs):
        shape = tuple(shapes[key])
        if key.endswith("A_log"):
            value = jnp.zeros(shape, dtype)
        elif key.endswith("dt_bias") or key.endswith("running_var"):
            value = jnp.ones(shape, dtype)
        elif key.endswith("running_mean") or key.endswith(".bias"):
            value = jnp.zeros(shape, dtype)
        elif _is_norm_weight(key, shape):
            value = jnp.ones(shape, dtype)
        else:
            fan_in = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
            value = jax.random.normal(sub, shape, dtype=dtype) * (1.0 / np.sqrt(max(fan_in, 1)))
        flat[key] = value
    return weights_mod.unflatten_flat(flat, list_containers)
