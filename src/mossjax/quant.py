import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

PRECISION = lax.Precision.HIGHEST
INT8_MAX = 127.0
INV_INT8_MAX = np.float32(1.0) / np.float32(INT8_MAX)


def quantize_weight_host(weight):
    array = np.asarray(weight, dtype=np.float32)
    scale = np.maximum(np.max(np.abs(array), axis=1), np.float32(1e-8)) * INV_INT8_MAX
    quantized = np.clip(np.round(array / scale[:, None]), -INT8_MAX, INT8_MAX).astype(np.int8)
    return quantized, scale.astype(np.float32)


def host_layer(weight):
    quantized, scale = quantize_weight_host(weight)
    return {
        "int8": quantized,
        "scale": scale,
        "act_dtype": np.zeros((), dtype=np.asarray(weight).dtype),
    }


def leaf_from_host(value):
    act = value.get("act_dtype")
    act_dtype = jnp.asarray(act).dtype if act is not None else jnp.float32
    return {
        "int8": jnp.asarray(value["int8"]),
        "scale": jnp.asarray(value["scale"]),
        "act_dtype": jnp.zeros((), dtype=act_dtype),
    }


def per_output_channel_scale(weight, eps=1e-8):
    weight = weight.astype(jnp.float32)
    return jnp.maximum(jnp.max(jnp.abs(weight), axis=1), jnp.asarray(eps, jnp.float32)) * jnp.asarray(INV_INT8_MAX)


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


def quantize_text_encoder_params(params):
    out = dict(params)
    out["embed_tokens"] = {"weight": quantize_linear_layer(params["embed_tokens"]["weight"])}
    layers = []
    for layer in params["layers"]:
        new_layer = dict(layer)
        if "mlp" in layer:
            new_layer["mlp"] = {
                name: {"weight": quantize_linear_layer(layer["mlp"][name]["weight"])}
                for name in ("gate_proj", "up_proj", "down_proj")
            }
        if "self_attn" in layer:
            attention = dict(layer["self_attn"])
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                attention[name] = {"weight": quantize_linear_layer(layer["self_attn"][name]["weight"])}
            new_layer["self_attn"] = attention
        else:
            linear = dict(layer["linear_attn"])
            for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
                linear[name] = {"weight": quantize_linear_layer(layer["linear_attn"][name]["weight"])}
            new_layer["linear_attn"] = linear
        layers.append(new_layer)
    out["layers"] = layers
    return out


def tree_nbytes(tree):
    return sum(getattr(leaf, "nbytes", 0) for leaf in jax.tree.leaves(tree))
