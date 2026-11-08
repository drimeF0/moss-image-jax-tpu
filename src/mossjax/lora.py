import math
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from . import weights as weights_mod
from .config import MossConfig

UNET_PREFIX = "lora_unet"
TARGET_SUFFIXES = ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down")


def target_paths(cfg: MossConfig):
    return [f"layers.{index}.{suffix}" for index in range(cfg.num_layers) for suffix in TARGET_SUFFIXES]


def module_name(path: str) -> str:
    return f"{UNET_PREFIX}_{path}".replace(".", "_")


def _path_map(cfg: MossConfig):
    return {module_name(path): path for path in target_paths(cfg)}


def _suffix_name(suffix: str) -> str:
    return suffix.split(".")[1]


def _weight_shape(leaf):
    if isinstance(leaf, dict):
        return tuple(leaf["int8"].shape)
    return tuple(leaf.shape)


def init_lora(params, cfg: MossConfig, rank: int = 4, alpha=None, rng=None, dtype=jnp.float32):
    if alpha is None:
        alpha = float(rank)
    if rng is None:
        rng = jax.random.PRNGKey(0)
    scale = jnp.asarray(float(alpha) / rank, dtype=dtype)
    layers = []
    for index in range(cfg.num_layers):
        projections = {}
        for target_index, suffix in enumerate(TARGET_SUFFIXES):
            name = _suffix_name(suffix)
            out_dim, in_dim = _weight_shape(params["layers"][index]["proj"][name]["weight"])
            key = jax.random.fold_in(rng, index * len(TARGET_SUFFIXES) + target_index)
            bound = 1.0 / math.sqrt(in_dim)
            down = jax.random.uniform(key, (rank, in_dim), minval=-bound, maxval=bound, dtype=dtype)
            up = jnp.zeros((out_dim, rank), dtype=dtype)
            projections[name] = {"A": down, "B": up, "scale": scale}
        layers.append({"proj": projections})
    return {"layers": layers}


def merged_weight(weight, lora_leaf):
    return weight + lora_leaf["scale"] * (lora_leaf["B"] @ lora_leaf["A"])


def export_lora(lora, cfg: MossConfig, dim, alpha, dtype=np.float32) -> Dict[str, np.ndarray]:
    flat = {}
    for index, block in enumerate(lora["layers"]):
        for suffix in TARGET_SUFFIXES:
            leaf = block["proj"][_suffix_name(suffix)]
            name = module_name(f"layers.{index}.{suffix}")
            flat[name + ".lora_down.weight"] = np.array(leaf["A"], dtype=dtype)
            flat[name + ".lora_up.weight"] = np.array(leaf["B"], dtype=dtype)
            flat[name + ".alpha"] = np.array(alpha, dtype=np.float32)
    return flat


def export_metadata(dim, alpha, module: str = "networks.lora_generic"):
    return {
        "ss_network_module": module,
        "ss_network_dim": str(dim),
        "ss_network_alpha": str(alpha),
    }


def save_lora_file(path, lora, cfg: MossConfig, dim, alpha, metadata=None, dtype=np.float32):
    from safetensors.numpy import save_file

    flat = export_lora(lora, cfg, dim, alpha, dtype)
    header = export_metadata(dim, alpha)
    if metadata is not None:
        header.update({str(key): str(value) for key, value in metadata.items()})
    save_file(flat, str(path), metadata=header)


def load_lora_file(path, params, cfg: MossConfig, dtype=np.float32):
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as handle:
        sd = {key: handle.get_tensor(key) for key in handle.keys()}
    return import_lora(sd, params, cfg, dtype)


def import_lora(sd: Dict[str, np.ndarray], params, cfg: MossConfig, dtype=jnp.float32):
    path_map = _path_map(cfg)
    collected = {}
    for key, value in sd.items():
        module, _, tail = key.partition(".")
        if module not in path_map:
            raise ValueError(f"unknown LoRA module in weights: {module}")
        leaf = collected.setdefault(path_map[module], {})
        array = np.asarray(value)
        if tail == "lora_down.weight":
            leaf["A"] = jnp.asarray(array, dtype=dtype)
        elif tail == "lora_up.weight":
            leaf["B"] = jnp.asarray(array, dtype=dtype)
        elif tail == "alpha":
            leaf["alpha"] = float(array)
        else:
            raise ValueError(f"unexpected LoRA key: {key}")

    flat = weights_mod.flatten_tree(params)
    layers = []
    for index in range(cfg.num_layers):
        projections = {}
        for suffix in TARGET_SUFFIXES:
            path = f"layers.{index}.{suffix}"
            leaf = collected.get(path)
            if leaf is None or "A" not in leaf or "B" not in leaf:
                raise ValueError(f"missing LoRA weights for {path}")
            rank = leaf["A"].shape[0]
            alpha = leaf.get("alpha", float(rank))
            projections[_suffix_name(suffix)] = {
                "A": leaf["A"],
                "B": leaf["B"],
                "scale": jnp.asarray(alpha / rank, dtype=dtype),
            }
        layers.append({"proj": projections})
    return {"layers": layers}


def lora_rank_alpha(lora_tree, default_rank: int = 4, default_alpha=None):
    try:
        leaf = lora_tree["layers"][0]["proj"]["q"]
        rank = int(leaf["A"].shape[0])
        scale = float(np.asarray(leaf["scale"]))
        alpha = scale * rank
        if alpha <= 0:
            raise ValueError("non-positive alpha")
        return rank, alpha
    except Exception:
        return default_rank, (default_alpha if default_alpha is not None else float(default_rank))


def merge_into_params(params, lora, cfg: MossConfig, dtype=None):
    flat = weights_mod.flatten_tree(params)
    for index, block in enumerate(lora["layers"]):
        for suffix in TARGET_SUFFIXES:
            path = f"layers.{index}.{suffix}.weight"
            weight = flat[path]
            target_dtype = weight.dtype if dtype is None else dtype
            merged = merged_weight(weight.astype(jnp.float32), block["proj"][_suffix_name(suffix)])
            flat[path] = merged.astype(target_dtype)
    return weights_mod.unflatten_flat(flat)
