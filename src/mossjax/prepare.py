import os
import time

import numpy as np
from safetensors.numpy import save_file

from . import quant, text_encoder as text_encoder_mod, weights
from .config import MOSS_IMAGE_CONFIG


def _iter_shards(paths):
    from safetensors import safe_open

    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        with safe_open(str(path), framework="np") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)


def _merge_dtype(previous, current):
    dtype = np.dtype(current)
    if previous is None:
        return dtype
    if dtype != previous:
        raise ValueError(f"mixed quantization dtypes in the checkpoint: {previous} and {dtype}")
    return previous


def _finalize(outputs, output_path, act_dtype, quantized, started, source_paths):
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    metadata = {quant.META_FORMAT: quant.FORMAT_INT8_SCALE}
    if act_dtype is not None:
        metadata[quant.META_ACT_DTYPE] = quant.dtype_name(act_dtype)
    save_file(outputs, str(output_path), metadata=metadata)
    source_bytes = sum(os.path.getsize(path) for path in source_paths)
    return {
        "path": str(output_path),
        "keys": len(outputs),
        "quantized_keys": quantized,
        "act_dtype": quant.dtype_name(act_dtype) if act_dtype is not None else None,
        "bytes": os.path.getsize(output_path),
        "source_bytes": source_bytes,
        "seconds": round(time.time() - started, 2),
    }


def prepare_dit(source_paths, output_path, cfg=MOSS_IMAGE_CONFIG):
    started = time.time()
    outputs, quantized, act_dtype = {}, 0, None
    for key, tensor in _iter_shards(source_paths):
        key = weights.strip_dit_prefix(key)
        if key in weights.DERIVED_KEYS:
            continue
        split = weights.split_fused_qkv(key, cfg.dim)
        if split is not None:
            array = np.asarray(tensor)
            if array.shape[0] != 3 * cfg.dim:
                raise ValueError(f"fused qkv tensor has shape {array.shape}, expected first dim {3 * cfg.dim}")
            pairs = [(new_key, np.ascontiguousarray(part)) for new_key, part in zip(split, np.split(array, 3, axis=0))]
        else:
            pairs = [(key, np.asarray(tensor))]
        for new_key, array in pairs:
            if weights.is_quant_target(new_key):
                quantized_weight, scale = quant.quantize_weight_host(array)
                outputs[new_key] = quantized_weight
                outputs[new_key + ".scale"] = scale
                quantized += 1
                act_dtype = _merge_dtype(act_dtype, array.dtype)
            else:
                outputs[new_key] = array
    return _finalize(outputs, output_path, act_dtype, quantized, started, list(source_paths))


def prepare_text_encoder(source_paths, te_cfg, output_path):
    started = time.time()
    expected = text_encoder_mod.param_shapes(te_cfg)
    outputs, quantized, act_dtype = {}, 0, None
    for key, tensor in _iter_shards(source_paths):
        key = text_encoder_mod.strip_text_encoder_key(key)
        if key is None or key not in expected:
            continue
        array = np.asarray(tensor)
        if tuple(array.shape) != tuple(expected[key]):
            raise ValueError(f"shape mismatch for {key}: {array.shape} != {expected[key]}")
        if text_encoder_mod.is_quant_target(key):
            quantized_weight, scale = quant.quantize_weight_host(array)
            outputs[key] = quantized_weight
            outputs[key + ".scale"] = scale
            quantized += 1
            act_dtype = _merge_dtype(act_dtype, array.dtype)
        else:
            outputs[key] = array
    return _finalize(outputs, output_path, act_dtype, quantized, started, list(source_paths))
