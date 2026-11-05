from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from .config import MossConfig

DIT_KEY_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
DERIVED_KEYS = ("rope.bands",)


def strip_dit_prefix(key: str) -> str:
    for prefix in DIT_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def split_fused_qkv(key: str, dim: int):
    if not key.startswith("layers."):
        return None
    suffix = None
    for candidate in (".proj.qkv.weight", ".proj.qkv.weight_scale", ".proj.qkv.comfy_quant"):
        if key.endswith(candidate):
            suffix = candidate[len(".proj.qkv"):]
            break
    if suffix is None:
        return None
    base = key[: -len("qkv" + suffix)]
    return [base + "q" + suffix, base + "k" + suffix, base + "v" + suffix]


LIST_CONTAINERS = ("layers", "blocks")


def _parse_key(key: str):
    parts = key.split(".")
    return [int(segment) if index > 0 and parts[index - 1] in LIST_CONTAINERS else segment for index, segment in enumerate(parts)]


def _assign(tree, parts, value):
    key = parts[0]
    if len(parts) == 1:
        tree[key] = value
        return
    child = tree.get(key)
    if child is None:
        child = [] if isinstance(parts[1], int) else {}
        tree[key] = child
    if isinstance(child, list):
        index = parts[1]
        while len(child) <= index:
            child.append(None)
        if len(parts) == 2:
            child[index] = value
        else:
            if child[index] is None:
                child[index] = [] if isinstance(parts[2], int) else {}
            _assign(child[index], parts[2:], value)
    else:
        _assign(child, parts[1:], value)


def unflatten_flat(flat: Dict[str, jnp.ndarray]):
    tree = {}
    for key, value in flat.items():
        _assign(tree, _parse_key(key), value)
    return tree


def flatten_tree(tree, prefix=""):
    flat = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            flat.update(flatten_tree(value, f"{prefix}{key}."))
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            flat.update(flatten_tree(value, f"{prefix}{index}."))
    else:
        flat[prefix[:-1]] = tree
    return flat


def param_shapes(cfg: MossConfig):
    head_dim = cfg.head_dim
    dim, ffn_dim = cfg.dim, cfg.ffn_dim
    shapes = {
        "video_embedder.weight": (dim, cfg.latent_channels),
        "video_embedder.bias": (dim,),
        "text_embedder.proj.weight": (dim, cfg.text_dim),
        "text_embedder.proj.bias": (dim,),
        "time_embedding.0.weight": (dim, cfg.freq_dim),
        "time_embedding.0.bias": (dim,),
        "time_embedding.2.weight": (dim, dim),
        "time_embedding.2.bias": (dim,),
        "shared_block_modulation.weight": (6 * dim, dim),
        "shared_block_modulation.bias": (6 * dim,),
        "final_norm_video.weight": (dim,),
        "final_linear_video.weight": (cfg.latent_channels, dim),
        "video_head_modulation.weight": (2 * dim, dim),
        "video_head_modulation.bias": (2 * dim,),
    }
    for i in range(2):
        prefix = f"text_embedder.blocks.{i}."
        shapes.update(
            {
                prefix + "norm1.weight": (dim,),
                prefix + "norm2.weight": (dim,),
                prefix + "q_norm.weight": (head_dim,),
                prefix + "k_norm.weight": (head_dim,),
                prefix + "qkv.weight": (3 * dim, dim),
                prefix + "qkv.bias": (3 * dim,),
                prefix + "proj.weight": (dim, dim),
                prefix + "proj.bias": (dim,),
                prefix + "ffn_gate.weight": (ffn_dim, dim),
                prefix + "ffn_gate.bias": (ffn_dim,),
                prefix + "ffn_up.weight": (ffn_dim, dim),
                prefix + "ffn_up.bias": (ffn_dim,),
                prefix + "ffn_down.weight": (dim, ffn_dim),
                prefix + "ffn_down.bias": (dim,),
            }
        )
    for i in range(cfg.num_layers):
        prefix = f"layers.{i}."
        shapes.update(
            {
                prefix + "norm1.weight": (dim,),
                prefix + "norm2.weight": (dim,),
                prefix + "q_norm.weight": (head_dim,),
                prefix + "k_norm.weight": (head_dim,),
                prefix + "proj.q.weight": (dim, dim),
                prefix + "proj.k.weight": (dim, dim),
                prefix + "proj.v.weight": (dim, dim),
                prefix + "proj.o.weight": (dim, dim),
                prefix + "proj.ffn_up_gate.weight": (2 * ffn_dim, dim),
                prefix + "proj.ffn_down.weight": (dim, ffn_dim),
            }
        )
    return shapes


def state_dict_to_params(sd: Dict[str, np.ndarray], cfg: MossConfig):
    expected = param_shapes(cfg)
    flat = {}
    for key, tensor in sd.items():
        key = strip_dit_prefix(key)
        if key in DERIVED_KEYS:
            continue
        split = split_fused_qkv(key, cfg.dim)
        if split is not None:
            array = np.asarray(tensor)
            if array.shape[0] != 3 * cfg.dim:
                raise ValueError(f"fused qkv tensor has shape {array.shape}, expected first dim {3 * cfg.dim}")
            for new_key, part in zip(split, np.split(array, 3, axis=0)):
                flat[new_key] = jnp.asarray(np.ascontiguousarray(part))
            continue
        flat[key] = jnp.asarray(np.asarray(tensor))

    missing = sorted(set(expected) - set(flat))
    unexpected = sorted(set(flat) - set(expected))
    if missing:
        raise ValueError(f"missing keys in state dict: {missing[:10]}")
    if unexpected:
        raise ValueError(f"unexpected keys in state dict: {unexpected[:10]}")
    for key, shape in expected.items():
        if tuple(flat[key].shape) != tuple(shape):
            raise ValueError(f"shape mismatch for {key}: {flat[key].shape} != {shape}")
    return unflatten_flat(flat)


def params_to_state_dict(params) -> Dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in flatten_tree(params).items()}


def tree_dtypes(params):
    return {key: value.dtype for key, value in flatten_tree(params).items()}


def cast_tree(params, dtype):
    return jax.tree.map(lambda leaf: leaf.astype(dtype), params)
