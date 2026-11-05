import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import weights as weights_mod
from .quant import PRECISION

LATENT_CHANNELS = 32
LATENT_SCALE_FACTOR = 8
PATCH_SIZE = 2
PATCHIFIED_LATENT_CHANNELS = LATENT_CHANNELS * PATCH_SIZE * PATCH_SIZE
BN_EPS = 1e-4
VAE_LIST_CONTAINERS = ("down", "up", "block")


@dataclass(frozen=True)
class VaeConfig:
    ch: int = 128
    decoder_ch: Optional[int] = None
    z_channels: int = 32
    ch_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    num_groups: int = 32
    in_channels: int = 3
    out_channels: int = 3
    conv_eps: float = 1e-6
    bn_eps: float = BN_EPS


def conv2d(x, weight, bias=None, stride=1, padding=((1, 1), (1, 1))):
    y = lax.conv_general_dilated(
        x,
        weight,
        (stride, stride),
        padding,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        precision=PRECISION,
    )
    if bias is not None:
        y = y + bias[None, :, None, None]
    return y


def group_norm(x, weight, bias, num_groups=32, eps=1e-6):
    batch, channels, height, width = x.shape
    grouped = x.reshape(batch, num_groups, channels // num_groups, height, width)
    mean = jnp.mean(grouped, axis=(2, 3, 4), keepdims=True)
    var = jnp.mean(jnp.square(grouped - mean), axis=(2, 3, 4), keepdims=True)
    normalized = ((grouped - mean) * lax.rsqrt(var + eps)).reshape(batch, channels, height, width)
    return normalized * weight[None, :, None, None] + bias[None, :, None, None]


def swish(x):
    return x * jax.nn.sigmoid(x)


def attn_block(x, params, cfg):
    h = group_norm(x, params["norm"]["weight"], params["norm"]["bias"], cfg.num_groups, cfg.conv_eps)
    q = conv2d(h, params["q"]["weight"], params["q"]["bias"], 1, ((0, 0), (0, 0)))
    k = conv2d(h, params["k"]["weight"], params["k"]["bias"], 1, ((0, 0), (0, 0)))
    v = conv2d(h, params["v"]["weight"], params["v"]["bias"], 1, ((0, 0), (0, 0)))
    batch, channels, height, width = q.shape
    length = height * width
    q = q.reshape(batch, 1, channels, length).transpose(0, 1, 3, 2)
    k = k.reshape(batch, 1, channels, length).transpose(0, 1, 3, 2)
    v = v.reshape(batch, 1, channels, length).transpose(0, 1, 3, 2)
    scale = 1.0 / math.sqrt(channels)
    scores = jnp.einsum("bqld,bqkd->bqlk", q, k, precision=PRECISION) * scale
    probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
    out = jnp.einsum("bqlk,bqkd->bqld", probs, v, precision=PRECISION)
    out = out.transpose(0, 1, 3, 2).reshape(batch, channels, height, width)
    return x + conv2d(out, params["proj_out"]["weight"], params["proj_out"]["bias"], 1, ((0, 0), (0, 0)))


def resnet_block(x, params, cfg):
    h = swish(group_norm(x, params["norm1"]["weight"], params["norm1"]["bias"], cfg.num_groups, cfg.conv_eps))
    h = conv2d(h, params["conv1"]["weight"], params["conv1"]["bias"], 1, ((1, 1), (1, 1)))
    h = swish(group_norm(h, params["norm2"]["weight"], params["norm2"]["bias"], cfg.num_groups, cfg.conv_eps))
    h = conv2d(h, params["conv2"]["weight"], params["conv2"]["bias"], 1, ((1, 1), (1, 1)))
    if "nin_shortcut" in params:
        x = conv2d(x, params["nin_shortcut"]["weight"], params["nin_shortcut"]["bias"], 1, ((0, 0), (0, 0)))
    return x + h


def downsample(x, params):
    x = jnp.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)))
    return conv2d(x, params["conv"]["weight"], params["conv"]["bias"], 2, ((0, 0), (0, 0)))


def upsample(x, params):
    x = jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)
    return conv2d(x, params["conv"]["weight"], params["conv"]["bias"], 1, ((1, 1), (1, 1)))


def encoder_forward(x, params, cfg):
    h = conv2d(x, params["conv_in"]["weight"], params["conv_in"]["bias"], 1, ((1, 1), (1, 1)))
    num_levels = len(cfg.ch_mult)
    for level in range(num_levels):
        for block in params["down"][level]["block"]:
            h = resnet_block(h, block, cfg)
        if level != num_levels - 1:
            h = downsample(h, params["down"][level]["downsample"])
    h = resnet_block(h, params["mid"]["block_1"], cfg)
    h = attn_block(h, params["mid"]["attn_1"], cfg)
    h = resnet_block(h, params["mid"]["block_2"], cfg)
    h = swish(group_norm(h, params["norm_out"]["weight"], params["norm_out"]["bias"], cfg.num_groups, cfg.conv_eps))
    h = conv2d(h, params["conv_out"]["weight"], params["conv_out"]["bias"], 1, ((1, 1), (1, 1)))
    return conv2d(h, params["quant_conv"]["weight"], params["quant_conv"]["bias"], 1, ((0, 0), (0, 0)))


def decoder_forward(z, params, cfg):
    z = conv2d(z, params["post_quant_conv"]["weight"], params["post_quant_conv"]["bias"], 1, ((0, 0), (0, 0)))
    h = conv2d(z, params["conv_in"]["weight"], params["conv_in"]["bias"], 1, ((1, 1), (1, 1)))
    h = resnet_block(h, params["mid"]["block_1"], cfg)
    h = attn_block(h, params["mid"]["attn_1"], cfg)
    h = resnet_block(h, params["mid"]["block_2"], cfg)
    for level in reversed(range(len(cfg.ch_mult))):
        for block in params["up"][level]["block"]:
            h = resnet_block(h, block, cfg)
        if level != 0:
            h = upsample(h, params["up"][level]["upsample"])
    h = swish(group_norm(h, params["norm_out"]["weight"], params["norm_out"]["bias"], cfg.num_groups, cfg.conv_eps))
    return conv2d(h, params["conv_out"]["weight"], params["conv_out"]["bias"], 1, ((1, 1), (1, 1)))


def patchify(z):
    batch, channels, height, width = z.shape
    return (
        z.reshape(batch, channels, height // PATCH_SIZE, PATCH_SIZE, width // PATCH_SIZE, PATCH_SIZE)
        .transpose(0, 1, 3, 5, 2, 4)
        .reshape(batch, channels * PATCH_SIZE * PATCH_SIZE, height // PATCH_SIZE, width // PATCH_SIZE)
    )


def unpatchify(z):
    batch, channels4, height, width = z.shape
    channels = channels4 // (PATCH_SIZE * PATCH_SIZE)
    return (
        z.reshape(batch, channels, PATCH_SIZE, PATCH_SIZE, height, width)
        .transpose(0, 1, 4, 2, 5, 3)
        .reshape(batch, channels, height * PATCH_SIZE, width * PATCH_SIZE)
    )


def normalize(z, running_mean, running_var, eps=BN_EPS):
    scale = jnp.sqrt(running_var.astype(z.dtype) + eps).reshape(1, -1, 1, 1)
    mean = running_mean.astype(z.dtype).reshape(1, -1, 1, 1)
    return (z - mean) / scale


def denormalize(z, running_mean, running_var, eps=BN_EPS):
    scale = jnp.sqrt(running_var.astype(z.dtype) + eps).reshape(1, -1, 1, 1)
    mean = running_mean.astype(z.dtype).reshape(1, -1, 1, 1)
    return z * scale + mean


def encode(params, x, cfg):
    return jnp.split(encoder_forward(x, params["encoder"], cfg), 2, axis=1)[0]


def decode(params, z, cfg):
    return decoder_forward(z, params["decoder"], cfg)


def encode_for_dit(params, x, cfg):
    return normalize(
        patchify(encode(params, x, cfg)),
        params["bn"]["running_mean"],
        params["bn"]["running_var"],
        cfg.bn_eps,
    )


def decode_from_dit(params, z, cfg):
    return decode(
        params,
        unpatchify(denormalize(z, params["bn"]["running_mean"], params["bn"]["running_var"], cfg.bn_eps)),
        cfg,
    )


def _add_conv(shapes, prefix, out_channels, in_channels, kernel=3, bias=True):
    shapes[f"{prefix}.weight"] = (out_channels, in_channels, kernel, kernel)
    if bias:
        shapes[f"{prefix}.bias"] = (out_channels,)


def _add_resnet(shapes, prefix, in_channels, out_channels, with_shortcut):
    shapes[f"{prefix}.norm1.weight"] = (in_channels,)
    shapes[f"{prefix}.norm1.bias"] = (in_channels,)
    _add_conv(shapes, f"{prefix}.conv1", out_channels, in_channels)
    shapes[f"{prefix}.norm2.weight"] = (out_channels,)
    shapes[f"{prefix}.norm2.bias"] = (out_channels,)
    _add_conv(shapes, f"{prefix}.conv2", out_channels, out_channels)
    if with_shortcut:
        _add_conv(shapes, f"{prefix}.nin_shortcut", out_channels, in_channels, kernel=1)


def _add_attn(shapes, prefix, channels):
    shapes[f"{prefix}.norm.weight"] = (channels,)
    shapes[f"{prefix}.norm.bias"] = (channels,)
    for name in ("q", "k", "v", "proj_out"):
        _add_conv(shapes, f"{prefix}.{name}", channels, channels, kernel=1)


def param_shapes(cfg: VaeConfig):
    shapes = {}
    encoder_ch = cfg.ch
    levels = cfg.ch_mult
    num_levels = len(levels)

    _add_conv(shapes, "encoder.conv_in", encoder_ch, cfg.in_channels)
    block_in = encoder_ch
    for level in range(num_levels):
        block_out = encoder_ch * levels[level]
        for index in range(cfg.num_res_blocks):
            in_channels = block_in if index == 0 else block_out
            _add_resnet(shapes, f"encoder.down.{level}.block.{index}", in_channels, block_out, in_channels != block_out)
        block_in = block_out
        if level != num_levels - 1:
            _add_conv(shapes, f"encoder.down.{level}.downsample.conv", block_in, block_in)
    _add_resnet(shapes, "encoder.mid.block_1", block_in, block_in, False)
    _add_attn(shapes, "encoder.mid.attn_1", block_in)
    _add_resnet(shapes, "encoder.mid.block_2", block_in, block_in, False)
    shapes["encoder.norm_out.weight"] = (block_in,)
    shapes["encoder.norm_out.bias"] = (block_in,)
    _add_conv(shapes, "encoder.conv_out", 2 * cfg.z_channels, block_in)
    _add_conv(shapes, "encoder.quant_conv", 2 * cfg.z_channels, 2 * cfg.z_channels, kernel=1)

    decoder_ch = cfg.decoder_ch if cfg.decoder_ch is not None else cfg.ch
    block_in = decoder_ch * levels[-1]
    _add_conv(shapes, "decoder.conv_in", block_in, cfg.z_channels)
    _add_resnet(shapes, "decoder.mid.block_1", block_in, block_in, False)
    _add_attn(shapes, "decoder.mid.attn_1", block_in)
    _add_resnet(shapes, "decoder.mid.block_2", block_in, block_in, False)
    for level in reversed(range(num_levels)):
        block_out = decoder_ch * levels[level]
        for index in range(cfg.num_res_blocks + 1):
            in_channels = block_in if index == 0 else block_out
            _add_resnet(shapes, f"decoder.up.{level}.block.{index}", in_channels, block_out, in_channels != block_out)
        block_in = block_out
        if level != 0:
            _add_conv(shapes, f"decoder.up.{level}.upsample.conv", block_in, block_in)
    shapes["decoder.norm_out.weight"] = (block_in,)
    shapes["decoder.norm_out.bias"] = (block_in,)
    _add_conv(shapes, "decoder.conv_out", cfg.out_channels, block_in)
    _add_conv(shapes, "decoder.post_quant_conv", cfg.z_channels, cfg.z_channels, kernel=1)

    shapes["bn.running_mean"] = (PATCHIFIED_LATENT_CHANNELS,)
    shapes["bn.running_var"] = (PATCHIFIED_LATENT_CHANNELS,)
    return shapes


def state_dict_to_params(sd: Dict[str, np.ndarray], cfg: VaeConfig):
    expected = param_shapes(cfg)
    flat = {}
    for key, tensor in sd.items():
        if key == "bn.num_batches_tracked":
            continue
        if key not in expected:
            raise ValueError(f"unexpected VAE key: {key}")
        array = np.asarray(tensor)
        if tuple(array.shape) != tuple(expected[key]):
            raise ValueError(f"shape mismatch for {key}: {array.shape} != {expected[key]}")
        flat[key] = jnp.asarray(array)
    missing = sorted(set(expected) - set(flat))
    if missing:
        raise ValueError(f"missing VAE keys: {missing[:10]}")
    return weights_mod.unflatten_flat(flat, VAE_LIST_CONTAINERS)


def convert_diffusers_state_dict(sd: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    new_sd = {}
    num_levels = len({key.split(".")[2] for key in sd if key.startswith("decoder.up_blocks.")})

    def convert_resnet(key):
        return key.replace("conv_shortcut.", "nin_shortcut.")

    for key, value in sd.items():
        new_key = key
        if key.startswith("quant_conv."):
            new_key = "encoder." + key
        elif key.startswith("post_quant_conv."):
            new_key = "decoder." + key
        elif ".mid_block." in key:
            prefix, rest = key.split(".mid_block.", 1)
            if rest.startswith("resnets."):
                _, index, rest2 = rest.split(".", 2)
                new_key = f"{prefix}.mid.block_{int(index) + 1}.{convert_resnet(rest2)}"
            elif rest.startswith("attentions."):
                _, index, rest2 = rest.split(".", 2)
                mapping = {
                    "group_norm": "norm",
                    "to_q": "q",
                    "to_k": "k",
                    "to_v": "v",
                    "to_out.0": "proj_out",
                }
                for source, target in mapping.items():
                    if rest2.startswith(source + "."):
                        rest2 = target + rest2[len(source):]
                        break
                new_key = f"{prefix}.mid.attn_1.{rest2}"
                if rest2.endswith(".weight") and np.asarray(value).ndim == 2:
                    value = np.asarray(value)[:, :, None, None]
        elif key.startswith("encoder.down_blocks."):
            _, _, index, rest = key.split(".", 3)
            if rest.startswith("resnets."):
                _, block_index, rest2 = rest.split(".", 2)
                new_key = f"encoder.down.{index}.block.{block_index}.{convert_resnet(rest2)}"
            elif rest.startswith("downsamplers."):
                _, _, rest2 = rest.split(".", 2)
                new_key = f"encoder.down.{index}.downsample.{rest2}"
        elif key.startswith("decoder.up_blocks."):
            _, _, index, rest = key.split(".", 3)
            level = num_levels - 1 - int(index)
            if rest.startswith("resnets."):
                _, block_index, rest2 = rest.split(".", 2)
                new_key = f"decoder.up.{level}.block.{block_index}.{convert_resnet(rest2)}"
            elif rest.startswith("upsamplers."):
                _, _, rest2 = rest.split(".", 2)
                new_key = f"decoder.up.{level}.upsample.{rest2}"
        elif key.startswith("encoder.conv_norm_out.") or key.startswith("decoder.conv_norm_out."):
            new_key = key.replace(".conv_norm_out.", ".norm_out.")
        new_sd[new_key] = value
    return new_sd


def is_diffusers_state_dict(sd: Dict[str, np.ndarray]) -> bool:
    return any(key.startswith("encoder.down_blocks.") for key in sd)


def load_vae_params(sd: Dict[str, np.ndarray], cfg: Optional[VaeConfig] = None):
    for prefix in ("vae.", "first_stage_model."):
        if any(key.startswith(prefix) for key in sd):
            sd = {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in sd.items()}
    if is_diffusers_state_dict(sd):
        sd = convert_diffusers_state_dict(sd)
    decoder_ch = np.asarray(sd["decoder.conv_out.weight"]).shape[1]
    cfg = VaeConfig() if cfg is None else cfg
    if decoder_ch != cfg.ch:
        cfg = VaeConfig(ch=cfg.ch, decoder_ch=decoder_ch)
    return state_dict_to_params(sd, cfg), cfg
