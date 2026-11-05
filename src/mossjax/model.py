import math

import jax
import jax.numpy as jnp
from jax import lax

from . import quant
from .config import MOSS_IMAGE_CONFIG, MossConfig

PRECISION = lax.Precision.HIGHEST


def weight_dtype(weight):
    if isinstance(weight, dict):
        return weight["act_dtype"].dtype
    return weight.dtype


def linear(x, weight, bias=None):
    if isinstance(weight, dict):
        if bias is not None:
            raise ValueError("quantized linear layers carry their own bias")
        return quant.weight_only_linear(x, weight)
    y = jnp.einsum("...i,oi->...o", x, weight, precision=PRECISION)
    if bias is not None:
        y = y + bias
    return y


def linear_with_lora(x, weight, lora_leaf):
    y = linear(x, weight)
    if lora_leaf is not None:
        inner = linear(x.astype(lora_leaf["A"].dtype), lora_leaf["A"])
        y = y + linear(inner, lora_leaf["B"]) * lora_leaf["scale"]
    return y


def rms_norm(x, weight, eps=1e-6):
    xf = x.astype(jnp.float32)
    y = xf * lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)
    return y.astype(x.dtype) * weight


def situ_glu(hidden, gate):
    return (4.0 * jnp.tanh(gate / 4.0) * jax.nn.sigmoid(gate)) * (25.0 * jnp.tanh(hidden / 25.0))


def rope_bands(head_dim, base):
    num_bands = head_dim // 8
    exponent = jnp.arange(num_bands, dtype=jnp.float32) / num_bands
    return 1.0 / (base**exponent)


def rope_embed(height, width, text_length, text_lens, bands):
    batch = text_lens.shape[0]
    h, w = jnp.meshgrid(
        jnp.arange(height, dtype=jnp.float32), jnp.arange(width, dtype=jnp.float32), indexing="ij"
    )
    image = jnp.stack((jnp.zeros_like(h), jnp.zeros_like(h), h, w), -1).reshape(-1, 4)
    image = image.at[:, 2].add(-(height - 1) / 2.0)
    image = image.at[:, 3].add(-(width - 1) / 2.0)
    image = jnp.broadcast_to(image[None], (batch, height * width, 4))
    image = image.at[:, :, 1].set(text_lens.astype(jnp.float32)[:, None])

    text = jnp.zeros((text_length, 4), dtype=jnp.float32)
    text = text.at[:, 1].set(jnp.arange(text_length, dtype=jnp.float32))
    text = jnp.broadcast_to(text[None], (batch, text_length, 4))

    phase = jnp.concatenate((image, text), axis=1)[..., None] * bands
    return jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-2).reshape(batch, height * width + text_length, -1)


def apply_rope(x, rope):
    sin, cos = jnp.split(rope, 2, axis=-1)
    sin = jnp.concatenate((sin, sin), -1)[..., None, :]
    cos = jnp.concatenate((cos, cos), -1)[..., None, :]
    a, b = jnp.split(x, 2, axis=-1)
    return x * cos + jnp.concatenate((-b, a), -1) * sin


def attention(q, k, v, key_mask=None):
    batch, length, heads, head_dim = q.shape
    qh = jnp.transpose(q, (0, 2, 1, 3))
    kh = jnp.transpose(k, (0, 2, 1, 3))
    vh = jnp.transpose(v, (0, 2, 1, 3))
    scale = 1.0 / math.sqrt(head_dim)
    scores = jnp.einsum("bhqd,bhkd->bhqk", qh, kh, precision=PRECISION) * scale
    if key_mask is not None:
        bias = jnp.where(key_mask[:, None, None, :], jnp.zeros((), scores.dtype), jnp.full((), -jnp.inf, scores.dtype))
        scores = scores + bias
    probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
    out = jnp.einsum("bhqk,bhkd->bhqd", probs, vh, precision=PRECISION)
    return jnp.transpose(out, (0, 2, 1, 3)).reshape(batch, length, heads * head_dim)


def get_time_modulation(timesteps, params, cfg):
    half = cfg.freq_dim // 2
    freq = 10000.0 ** (-jnp.arange(half, dtype=jnp.float64) / half)
    phase = timesteps.astype(jnp.float64).reshape(-1, 1) * freq
    embedding = jnp.concatenate((jnp.cos(phase), jnp.sin(phase)), axis=-1).astype(jnp.float32)

    w0 = params["time_embedding"]["0"]["weight"]
    time = linear(embedding.astype(w0.dtype), w0, params["time_embedding"]["0"]["bias"])
    time = jax.nn.silu(time)
    w2 = params["time_embedding"]["2"]["weight"]
    time = linear(time.astype(w2.dtype), w2, params["time_embedding"]["2"]["bias"]).astype(jnp.float32)

    sm = params["shared_block_modulation"]
    mod = linear(time.astype(sm["weight"].dtype), sm["weight"], sm["bias"]).reshape(-1, 6, cfg.dim)
    vm = params["video_head_modulation"]
    head = linear(time.astype(vm["weight"].dtype), vm["weight"], vm["bias"]).reshape(-1, 2, cfg.dim)
    return mod, head


def main_block(x, rope, modulation, params, key_mask, cfg, lora_block=None):
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = jnp.split(modulation, 6, axis=1)
    norm_dtype = params["norm1"]["weight"].dtype
    lora_proj = None if lora_block is None else lora_block["proj"]

    def lora_leaf(name):
        return None if lora_proj is None else lora_proj[name]

    y = rms_norm(x, params["norm1"]["weight"])
    y = (y * (1 + scale_a) + shift_a).astype(norm_dtype)

    batch, length, _ = y.shape
    q = linear_with_lora(y, params["proj"]["q"]["weight"], lora_leaf("q")).reshape(
        batch, length, cfg.num_heads, cfg.head_dim
    )
    k = linear_with_lora(y, params["proj"]["k"]["weight"], lora_leaf("k")).reshape(
        batch, length, cfg.num_heads, cfg.head_dim
    )
    v = linear_with_lora(y, params["proj"]["v"]["weight"], lora_leaf("v")).reshape(
        batch, length, cfg.num_heads, cfg.head_dim
    )
    q = rms_norm(q, params["q_norm"]["weight"])
    k = rms_norm(k, params["k_norm"]["weight"])
    q = apply_rope(q, rope).astype(v.dtype)
    k = apply_rope(k, rope).astype(v.dtype)

    attn = attention(q, k, v, key_mask)
    o = params["proj"]["o"]["weight"]
    x = x + linear_with_lora(attn.astype(weight_dtype(o)), o, lora_leaf("o")) * gate_a

    y = rms_norm(x, params["norm2"]["weight"])
    y = (y * (1 + scale_f) + shift_f).astype(norm_dtype)
    hidden, gate = jnp.split(linear_with_lora(y, params["proj"]["ffn_up_gate"]["weight"], lora_leaf("ffn_up_gate")), 2, axis=-1)
    x = x + linear_with_lora(situ_glu(hidden, gate), params["proj"]["ffn_down"]["weight"], lora_leaf("ffn_down")) * gate_f
    return x


def text_block(x, rope, params, key_mask, cfg):
    qkv = linear(rms_norm(x, params["norm1"]["weight"]), params["qkv"]["weight"], params["qkv"]["bias"])
    batch, length, _ = x.shape
    q, k, v = (t.reshape(batch, length, cfg.num_heads, cfg.head_dim) for t in jnp.split(qkv, 3, axis=-1))
    q = rms_norm(q, params["q_norm"]["weight"])
    k = rms_norm(k, params["k_norm"]["weight"])
    q = apply_rope(q, rope).astype(v.dtype)
    k = apply_rope(k, rope).astype(v.dtype)

    attn = attention(q, k, v, key_mask)
    x = x + linear(attn.astype(params["proj"]["weight"].dtype), params["proj"]["weight"], params["proj"]["bias"])

    y = rms_norm(x, params["norm2"]["weight"])
    hidden = linear(y, params["ffn_up"]["weight"], params["ffn_up"]["bias"])
    gate = linear(y, params["ffn_gate"]["weight"], params["ffn_gate"]["bias"])
    return x + linear(situ_glu(hidden, gate), params["ffn_down"]["weight"], params["ffn_down"]["bias"])


def text_embedder(context, rope_text, context_mask, params, cfg):
    p = params["text_embedder"]
    w = p["proj"]["weight"]
    x = linear(context.astype(w.dtype), w, p["proj"]["bias"])
    for block in p["blocks"]:
        x = text_block(x, rope_text, block, context_mask, cfg)
    return x


def make_checkpointed_block(cfg):
    def call(x, rope, modulation, block_params, key_mask, lora_block):
        return main_block(x, rope, modulation, block_params, key_mask, cfg, lora_block)

    return jax.checkpoint(call)


def forward(params, x, timesteps, context, context_mask=None, cfg=MOSS_IMAGE_CONFIG, skip_layers=(), lora=None, remat=False):
    batch, channels, height, width = x.shape
    num_img = height * width
    text_length = context.shape[1]
    if context_mask is None:
        context_mask = jnp.ones((batch, text_length), dtype=jnp.int32)
    context_mask = context_mask.astype(jnp.int32)
    text_lens = context_mask.sum(axis=1)
    skip = frozenset(skip_layers)

    mod, head = get_time_modulation(timesteps, params, cfg)
    bands = rope_bands(cfg.head_dim, cfg.rope_base)
    rope = rope_embed(height, width, text_length, text_lens, bands)

    vw = params["video_embedder"]
    video = x.reshape(batch, channels, num_img).transpose(0, 2, 1).astype(vw["weight"].dtype)
    video = linear(video, vw["weight"], vw["bias"])
    text = text_embedder(context, rope[:, num_img:], context_mask, params, cfg)
    hidden = jnp.concatenate((video, text), axis=1)

    key_mask = jnp.concatenate(
        (jnp.ones((batch, num_img), dtype=jnp.bool_), context_mask.astype(jnp.bool_)), axis=1
    )
    block_fn = make_checkpointed_block(cfg) if remat else None
    for index, block in enumerate(params["layers"]):
        if index in skip:
            continue
        lora_block = None if lora is None else lora["layers"][index]
        if remat:
            hidden = block_fn(hidden, rope, mod, block, key_mask, lora_block)
        else:
            hidden = main_block(hidden, rope, mod, block, key_mask, cfg, lora_block)

    hidden = hidden[:, :num_img]
    fnw = params["final_norm_video"]["weight"]
    hidden = rms_norm(hidden.astype(fnw.dtype), fnw)
    hidden = hidden * (1 + head[:, 1:2]) + head[:, 0:1]
    flw = params["final_linear_video"]["weight"]
    hidden = linear(hidden.astype(flw.dtype), flw)
    return hidden.transpose(0, 2, 1).reshape(batch, channels, height, width)
