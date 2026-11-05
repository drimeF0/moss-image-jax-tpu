import jax
import jax.numpy as jnp
from jax import lax

PRECISION = lax.Precision.HIGHEST
INT8_MAX = 127.0


def per_output_channel_scale(weight, eps=1e-8):
    weight = weight.astype(jnp.float32)
    return jnp.maximum(jnp.max(jnp.abs(weight), axis=1), eps) / INT8_MAX


def quantize_weight(weight):
    scale = per_output_channel_scale(weight)
    quantized = jnp.clip(jnp.round(weight.astype(jnp.float32) / scale[:, None]), -INT8_MAX, INT8_MAX).astype(jnp.int8)
    return quantized, scale.astype(jnp.float32)


def dequantize_weight(quantized, scale):
    return quantized.astype(jnp.float32) * scale[:, None]


def quantize_per_token(x, eps=1e-8):
    xf = x.astype(jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(xf), axis=-1, keepdims=True), eps) / INT8_MAX
    quantized = jnp.clip(jnp.round(xf / scale), -INT8_MAX, INT8_MAX).astype(jnp.int8)
    return quantized, scale.astype(jnp.float32)


def quantize_linear_layer(weight, bias=None):
    quantized, scale = quantize_weight(weight)
    layer = {"int8": quantized, "scale": scale, "act_dtype": jnp.zeros((), dtype=weight.dtype)}
    if bias is not None:
        layer["bias"] = bias
    return layer


def weight_only_linear(x, layer):
    weight = dequantize_weight(layer["int8"], layer["scale"]).astype(layer["act_dtype"].dtype)
    y = jnp.einsum("...i,oi->...o", x.astype(weight.dtype), weight, precision=PRECISION)
    if "bias" in layer:
        y = y + layer["bias"]
    return y


def int8_compute_linear(x, layer):
    x_q, x_scale = quantize_per_token(x)
    acc = lax.dot_general(
        x_q,
        layer["int8"],
        dimension_numbers=(((x_q.ndim - 1,), (1,)), ((), ())),
        preferred_element_type=jnp.int32,
    )
    y = acc.astype(jnp.float32) * (x_scale * layer["scale"])
    if "bias" in layer:
        y = y + layer["bias"].astype(jnp.float32)
    return y.astype(x.dtype)


def quantized_linear(x, layer, compute="bf16"):
    if compute == "bf16":
        y = weight_only_linear(x, layer)
    elif compute == "int8":
        y = int8_compute_linear(x, layer)
    else:
        raise ValueError(f"unknown compute mode {compute}")
    return y.astype(layer["act_dtype"].dtype)


def quantize_params(params, cfg):
    out = dict(params)
    out["layers"] = []
    for layer in params["layers"]:
        new_layer = dict(layer)
        new_proj = dict(layer["proj"])
        for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
            new_proj[name] = {"weight": quantize_linear_layer(layer["proj"][name]["weight"])}
        new_layer["proj"] = new_proj
        out["layers"].append(new_layer)
    return out


def tree_nbytes(tree):
    return sum(getattr(leaf, "nbytes", 0) for leaf in jax.tree.leaves(tree))
