from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from . import quant
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


DIT_LIST_CONTAINERS = ("layers", "blocks")


def _parse_key(key: str, list_containers=DIT_LIST_CONTAINERS):
    parts = key.split(".")
    return [
        int(segment) if index > 0 and parts[index - 1] in list_containers else segment
        for index, segment in enumerate(parts)
    ]


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


def unflatten_flat(flat: Dict[str, jnp.ndarray], list_containers=DIT_LIST_CONTAINERS):
    tree = {}
    for key, value in flat.items():
        _assign(tree, _parse_key(key, list_containers), value)
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


def is_quant_target(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    parts = key.split(".")
    return len(parts) == 5 and parts[0] == "layers" and parts[2] == "proj" and parts[3] in (
        "q",
        "k",
        "v",
        "o",
        "ffn_up_gate",
        "ffn_down",
    )


def _prepared_scales(sd):
    prepared = {}
    for key, value in sd.items():
        if key.endswith(".scale"):
            prepared[strip_dit_prefix(key[: -len(".scale")])] = value
    return prepared


def state_dict_to_params(sd: Dict[str, np.ndarray], cfg: MossConfig, quantize: bool = False, act_dtype=None):
    expected = param_shapes(cfg)
    prepared = _prepared_scales(sd)
    flat = {}
    for key, tensor in sd.items():
        if key.endswith(".scale"):
            continue
        key = strip_dit_prefix(key)
        if key in DERIVED_KEYS:
            continue
        if key in prepared and np.asarray(tensor).dtype == np.int8:
            flat[key] = {
                "int8": np.asarray(tensor),
                "scale": np.asarray(prepared[key]),
                "act_dtype": np.zeros((), dtype=act_dtype if act_dtype is not None else np.float32),
            }
            continue
        split = split_fused_qkv(key, cfg.dim)
        if split is not None:
            array = np.asarray(tensor)
            if array.shape[0] != 3 * cfg.dim:
                raise ValueError(f"fused qkv tensor has shape {array.shape}, expected first dim {3 * cfg.dim}")
            for new_key, part in zip(split, np.split(array, 3, axis=0)):
                flat[new_key] = np.ascontiguousarray(part)
            continue
        flat[key] = tensor if isinstance(tensor, dict) else np.asarray(tensor)

    missing = sorted(set(expected) - set(flat))
    unexpected = sorted(set(flat) - set(expected))
    if missing:
        raise ValueError(f"missing keys in state dict: {missing[:10]}")
    if unexpected:
        raise ValueError(f"unexpected keys in state dict: {unexpected[:10]}")

    converted = {}
    for key, value in flat.items():
        shape = value["int8"].shape if isinstance(value, dict) else value.shape
        if tuple(shape) != tuple(expected[key]):
            raise ValueError(f"shape mismatch for {key}: {shape} != {expected[key]}")
        if quantize and isinstance(value, np.ndarray) and is_quant_target(key):
            converted[key] = quant.leaf_from_host(quant.host_layer(value))
        elif isinstance(value, dict):
            converted[key] = quant.leaf_from_host(value)
        else:
            converted[key] = jnp.asarray(value)
    return unflatten_flat(converted)


def params_to_state_dict(params) -> Dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in flatten_tree(params).items()}


def tree_dtypes(params):
    return {key: value.dtype for key, value in flatten_tree(params).items()}


def cast_tree(params, dtype):
    return jax.tree.map(
        lambda leaf: leaf.astype(dtype) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, params
    )


def load_state_dict_from_safetensors(paths, disable_mmap: bool = False):
    from safetensors import safe_open

    if isinstance(paths, str):
        paths = [paths]
    sd = {}
    for path in paths:
        with safe_open(str(path), framework="np") as handle:
            for key in handle.keys():
                sd[key] = handle.get_tensor(key)
    return sd


def read_safetensors_metadata(paths):
    from safetensors import safe_open

    if isinstance(paths, str):
        paths = [paths]
    metadata = {}
    for path in paths:
        with safe_open(str(path), framework="np") as handle:
            metadata.update(handle.metadata() or {})
    return metadata


def _convert_leaf(key, array, prepared_scale, quantize, act_dtype):
    if prepared_scale is not None:
        return {
            "int8": jnp.asarray(array),
            "scale": jnp.asarray(prepared_scale),
            "act_dtype": jnp.zeros((), dtype=act_dtype if act_dtype is not None else np.float32),
        }
    if quantize and is_quant_target(key):
        return quant.leaf_from_host(quant.host_layer(array))
    return jnp.asarray(array)


def load_params(paths, cfg: MossConfig, dtype=None, quantize_int8: bool = False):
    from safetensors import safe_open

    if isinstance(paths, str):
        paths = [paths]
    metadata = read_safetensors_metadata(paths)
    act_dtype = quant.dtype_from_name(metadata.get(quant.META_ACT_DTYPE)) if metadata else None
    expected = param_shapes(cfg)

    prepared_scales = {}
    for path in paths:
        with safe_open(str(path), framework="np") as handle:
            for raw_key in handle.keys():
                if raw_key.endswith(".scale"):
                    prepared_scales[strip_dit_prefix(raw_key[: -len(".scale")])] = handle.get_tensor(raw_key)

    flat = {}
    for path in paths:
        with safe_open(str(path), framework="np") as handle:
            for raw_key in handle.keys():
                if raw_key.endswith(".scale"):
                    continue
                key = strip_dit_prefix(raw_key)
                if key in DERIVED_KEYS:
                    continue
                split = split_fused_qkv(key, cfg.dim)
                if split is not None:
                    array = np.asarray(handle.get_tensor(raw_key))
                    if array.shape[0] != 3 * cfg.dim:
                        raise ValueError(f"fused qkv tensor has shape {array.shape}, expected first dim {3 * cfg.dim}")
                    for new_key, part in zip(split, np.split(array, 3, axis=0)):
                        part = np.ascontiguousarray(part)
                        flat[new_key] = _convert_leaf(new_key, part, prepared_scales.get(new_key), quantize_int8, act_dtype)
                    del array
                    continue
                array = np.asarray(handle.get_tensor(raw_key))
                flat[key] = _convert_leaf(key, array, prepared_scales.get(key), quantize_int8, act_dtype)
                del array

    missing = sorted(set(expected) - set(flat))
    unexpected = sorted(set(flat) - set(expected))
    if missing:
        raise ValueError(f"missing keys in state dict: {missing[:10]}")
    if unexpected:
        raise ValueError(f"unexpected keys in state dict: {unexpected[:10]}")
    for key, value in flat.items():
        shape = value["int8"].shape if isinstance(value, dict) else value.shape
        if tuple(shape) != tuple(expected[key]):
            raise ValueError(f"shape mismatch for {key}: {shape} != {expected[key]}")

    params = unflatten_flat(flat)
    if dtype is not None:
        params = cast_tree(params, dtype)
    return params
