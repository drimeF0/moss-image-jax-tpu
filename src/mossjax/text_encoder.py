import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from . import quant
from . import weights as weights_mod
from .quant import PRECISION

TE_LIST_CONTAINERS = ("layers",)


@dataclass(frozen=True)
class TextConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: Tuple[int, ...] = (11, 11, 10)
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    layer_types: Tuple[str, ...] = field(default_factory=tuple)
    hidden_act: str = "silu"
    attention_bias: bool = False

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


def config_from_hf(config_dict: Dict) -> TextConfig:
    rope = config_dict.get("rope_parameters", {})
    layer_types = tuple(config_dict["layer_types"])
    return TextConfig(
        hidden_size=config_dict["hidden_size"],
        intermediate_size=config_dict["intermediate_size"],
        num_hidden_layers=config_dict["num_hidden_layers"],
        num_attention_heads=config_dict["num_attention_heads"],
        num_key_value_heads=config_dict["num_key_value_heads"],
        head_dim=config_dict["head_dim"],
        vocab_size=config_dict["vocab_size"],
        rms_norm_eps=config_dict["rms_norm_eps"],
        rope_theta=rope.get("rope_theta", config_dict.get("rope_theta", 10000000.0)),
        partial_rotary_factor=rope.get("partial_rotary_factor", config_dict.get("partial_rotary_factor", 1.0)),
        mrope_section=tuple(rope.get("mrope_section", config_dict.get("mrope_section", (11, 11, 10)))),
        linear_num_key_heads=config_dict["linear_num_key_heads"],
        linear_num_value_heads=config_dict["linear_num_value_heads"],
        linear_key_head_dim=config_dict["linear_key_head_dim"],
        linear_value_head_dim=config_dict["linear_value_head_dim"],
        linear_conv_kernel_dim=config_dict["linear_conv_kernel_dim"],
        layer_types=layer_types,
        hidden_act=config_dict.get("hidden_act", "silu"),
        attention_bias=config_dict.get("attention_bias", False),
    )


def _linear(x, weight):
    if isinstance(weight, dict):
        return quant.weight_only_linear(x, weight)
    return jnp.einsum("...i,oi->...o", x, weight, precision=PRECISION)


def embedding_lookup(params, input_ids):
    weight = params["embed_tokens"]["weight"]
    if isinstance(weight, dict):
        rows = jnp.take(weight["int8"], input_ids, axis=0).astype(jnp.float32)
        scale = jnp.take(weight["scale"], input_ids, axis=0)
        y = rows * scale[..., None]
        return y.astype(weight["act_dtype"].dtype)
    return jnp.take(weight, input_ids, axis=0)


def rms_norm_zero_centered(x, weight, eps):
    xf = x.astype(jnp.float32)
    y = xf * lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)
    return (y * (1.0 + weight.astype(jnp.float32))).astype(x.dtype)


def rms_norm_gated(hidden_states, gate, weight, eps):
    hidden_f = hidden_states.astype(jnp.float32)
    variance = jnp.mean(jnp.square(hidden_f), axis=-1, keepdims=True)
    normalized = hidden_f * lax.rsqrt(variance + eps)
    scaled = weight.astype(hidden_states.dtype) * normalized.astype(hidden_states.dtype)
    return (scaled * jax.nn.silu(gate.astype(jnp.float32))).astype(hidden_states.dtype)


def l2norm(x, eps=1e-6):
    return x * lax.rsqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + eps)


def rope_cos_sin(position_ids, cfg: TextConfig):
    dim = cfg.rotary_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freqs = position_ids.astype(jnp.float32)[..., None] * inv_freq
    return jnp.concatenate((jnp.cos(freqs), jnp.cos(freqs)), -1), jnp.concatenate((jnp.sin(freqs), jnp.sin(freqs)), -1)


def apply_rotary(x, cos, sin, cfg: TextConfig):
    rotary_dim = cfg.rotary_dim
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = jnp.concatenate((-x2, x1), axis=-1)
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return jnp.concatenate((x_rot * cos + rotated * sin, x_pass), axis=-1)


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, kv_heads, length, head_dim = hidden_states.shape
    expanded = jnp.broadcast_to(hidden_states[:, :, None], (batch, kv_heads, n_rep, length, head_dim))
    return expanded.reshape(batch, kv_heads * n_rep, length, head_dim)


def causal_bias(length, dtype):
    mask = jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))
    return jnp.where(mask, jnp.zeros((), dtype), jnp.full((), -jnp.inf, dtype))


def full_attention(x, params, cos, sin, cfg: TextConfig):
    batch, length, _ = x.shape
    q_gate = _linear(x, params["q_proj"]["weight"]).reshape(
        batch, length, cfg.num_attention_heads, 2 * cfg.head_dim
    )
    q, gate = q_gate[..., : cfg.head_dim], q_gate[..., cfg.head_dim :]
    gate = gate.reshape(batch, length, cfg.num_attention_heads * cfg.head_dim)
    k = _linear(x, params["k_proj"]["weight"]).reshape(
        batch, length, cfg.num_key_value_heads, cfg.head_dim
    )
    v = _linear(x, params["v_proj"]["weight"]).reshape(
        batch, length, cfg.num_key_value_heads, cfg.head_dim
    )
    q = rms_norm_zero_centered(q, params["q_norm"]["weight"], cfg.rms_norm_eps)
    k = rms_norm_zero_centered(k, params["k_norm"]["weight"], cfg.rms_norm_eps)

    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    q = apply_rotary(q, cos, sin, cfg)
    k = apply_rotary(k, cos, sin, cfg)
    k = repeat_kv(k, cfg.num_key_value_groups)
    v = repeat_kv(v, cfg.num_key_value_groups)

    scaling = cfg.head_dim**-0.5
    scores = jnp.einsum("bhld,bhkd->bhlk", q, k, precision=PRECISION) * scaling
    scores = scores + causal_bias(length, scores.dtype)
    probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
    out = jnp.einsum("bhlk,bhkd->bhld", probs, v, precision=PRECISION)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(batch, length, cfg.num_attention_heads * cfg.head_dim)
    out = out * jax.nn.sigmoid(gate)
    return _linear(out, params["o_proj"]["weight"])


def recurrent_gated_delta_rule(query, key, value, g, beta, cfg: TextConfig):
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)
    value = value.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    query = l2norm(query) * (query.shape[-1] ** -0.5)
    key = l2norm(key)

    def step(state, inputs):
        q_t, k_t, v_t, g_t, beta_t = inputs
        state = state * jnp.exp(g_t)[..., None, None]
        kv_mem = jnp.sum(state * k_t[..., None], axis=-2)
        delta = (v_t - kv_mem) * beta_t[..., None]
        state = state + k_t[..., None] * delta[..., None, :]
        out = jnp.sum(state * q_t[..., None], axis=-2)
        return state, out

    batch, length, heads, key_head_dim = key.shape
    value_head_dim = value.shape[-1]
    state = jnp.zeros((batch, heads, key_head_dim, value_head_dim), dtype=jnp.float32)
    inputs = (query.transpose(1, 0, 2, 3), key.transpose(1, 0, 2, 3), value.transpose(1, 0, 2, 3), g.transpose(1, 0, 2), beta.transpose(1, 0, 2))
    _, out = lax.scan(step, state, inputs)
    return jnp.transpose(out, (1, 0, 2, 3))


def gated_delta_net(x, params, cfg: TextConfig):
    batch, length, _ = x.shape
    mixed = _linear(x, params["in_proj_qkv"]["weight"])
    mixed = jnp.transpose(mixed, (0, 2, 1))
    pad = cfg.linear_conv_kernel_dim - 1
    mixed = jnp.pad(mixed, ((0, 0), (0, 0), (pad, 0)))
    mixed = lax.conv_general_dilated(
        mixed[:, :, None, :],
        params["conv1d"]["weight"][:, None],
        (1, 1),
        ((0, 0), (0, 0)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=cfg.conv_dim,
        precision=PRECISION,
    )[:, :, 0, : pad + length]
    mixed = jax.nn.silu(mixed)
    mixed = jnp.transpose(mixed, (0, 2, 1))

    query, key, value = jnp.split(mixed, [cfg.key_dim, 2 * cfg.key_dim], axis=-1)
    query = query.reshape(batch, length, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
    key = key.reshape(batch, length, cfg.linear_num_key_heads, cfg.linear_key_head_dim)
    value = value.reshape(batch, length, cfg.linear_num_value_heads, cfg.linear_value_head_dim)

    b = _linear(x, params["in_proj_b"]["weight"])
    a = _linear(x, params["in_proj_a"]["weight"])
    beta = jax.nn.sigmoid(b)
    g = -jnp.exp(params["A_log"].astype(jnp.float32)) * jax.nn.softplus(a.astype(jnp.float32) + params["dt_bias"])
    repeat = cfg.linear_num_value_heads // cfg.linear_num_key_heads
    if repeat > 1:
        query = jnp.repeat(query, repeat, axis=2)
        key = jnp.repeat(key, repeat, axis=2)

    core = recurrent_gated_delta_rule(query, key, value, g, beta, cfg)
    z = _linear(x, params["in_proj_z"]["weight"]).reshape(
        batch, length, cfg.linear_num_value_heads, cfg.linear_value_head_dim
    )
    core = rms_norm_gated(core.reshape(-1, cfg.linear_value_head_dim), z.reshape(-1, cfg.linear_value_head_dim), params["norm"]["weight"], cfg.rms_norm_eps)
    core = core.reshape(batch, length, cfg.value_dim)
    return _linear(core, params["out_proj"]["weight"])


def mlp(x, params, cfg: TextConfig):
    gate = _linear(x, params["gate_proj"]["weight"])
    up = _linear(x, params["up_proj"]["weight"])
    hidden = jax.nn.silu(gate) * up
    return _linear(hidden, params["down_proj"]["weight"])


def decoder_layer(x, params, cos, sin, cfg: TextConfig, layer_index):
    residual = x
    hidden = rms_norm_zero_centered(x, params["input_layernorm"]["weight"], cfg.rms_norm_eps)
    if cfg.layer_types[layer_index] == "full_attention":
        hidden = full_attention(hidden, params["self_attn"], cos, sin, cfg)
    else:
        hidden = gated_delta_net(hidden, params["linear_attn"], cfg)
    x = residual + hidden
    residual = x
    hidden = rms_norm_zero_centered(x, params["post_attention_layernorm"]["weight"], cfg.rms_norm_eps)
    return residual + mlp(hidden, params["mlp"], cfg)


def forward(params, input_ids, cfg: TextConfig, position_ids=None):
    input_ids = jnp.asarray(input_ids)
    if input_ids.ndim == 1:
        input_ids = input_ids[None]
    batch, length = input_ids.shape
    if position_ids is None:
        position_ids = jnp.broadcast_to(jnp.arange(length), (batch, length))
    cos, sin = rope_cos_sin(position_ids, cfg)
    x = embedding_lookup(params, input_ids)
    for index, layer in enumerate(params["layers"]):
        x = decoder_layer(x, layer, cos, sin, cfg, index)
    return rms_norm_zero_centered(x, params["norm"]["weight"], cfg.rms_norm_eps)


def state_dict_to_params(sd: Dict, cfg: TextConfig):
    flat = {key: jnp.asarray(value) for key, value in sd.items()}
    return weights_mod.unflatten_flat(flat, TE_LIST_CONTAINERS)


def load_config(path: str) -> TextConfig:
    with open(path, "r", encoding="utf-8") as handle:
        return config_from_hf(json.load(handle))


TEXT_ENCODER_SKIP_PREFIXES = ("visual.", "model.visual.", "mtp.", "lm_head.")
TEXT_ENCODER_KEY_PREFIXES = ("model.language_model.", "language_model.")


def strip_text_encoder_key(key: str):
    for prefix in TEXT_ENCODER_SKIP_PREFIXES:
        if key.startswith(prefix):
            return None
    for prefix in TEXT_ENCODER_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def load_state_dict_from_safetensors(paths):
    from safetensors import safe_open

    if isinstance(paths, str):
        paths = [paths]
    sd = {}
    for path in paths:
        with safe_open(str(path), framework="np") as handle:
            for key in handle.keys():
                stripped = strip_text_encoder_key(key)
                if stripped is not None:
                    sd[stripped] = handle.get_tensor(key)
    return sd


def load_params(paths, cfg: TextConfig, dtype=None):
    params = state_dict_to_params(load_state_dict_from_safetensors(paths), cfg)
    if dtype is not None:
        params = weights_mod.cast_tree(params, dtype)
    return params
