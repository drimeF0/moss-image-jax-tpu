import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

MOSS_LATENTS_SUFFIX = "_moss.npz"
MOSS_TE_SUFFIX = "_moss_te.npz"
LATENT_CHANNELS = 32
LATENT_STRIDE = 8


def float32_to_bf16_bits(x) -> np.ndarray:
    array = np.asarray(x)
    if array.dtype == np.uint16:
        return array
    bits = array.astype(np.float32).view(np.uint32)
    round_bit = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + round_bit
    return (rounded >> np.uint32(16)).astype(np.uint16)


def uint16_to_float32(bits) -> np.ndarray:
    array = np.asarray(bits)
    if array.dtype != np.uint16:
        array = array.astype(np.uint16)
    return (array.astype(np.uint32) << np.uint32(16)).view(np.float32)


def latents_npz_path(image_path: str, image_size: Tuple[int, int]) -> str:
    return os.path.splitext(image_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + MOSS_LATENTS_SUFFIX


def _latents_key_suffix(latents_shape) -> str:
    return f"_{latents_shape[-2]}x{latents_shape[-1]}"


def save_latents(path, latents, original_size, crop_ltrb, flipped_latents=None, alpha_mask=None):
    arrays = {}
    if os.path.exists(path):
        with np.load(path) as npz:
            arrays = {key: npz[key] for key in npz.files}
    suffix = _latents_key_suffix(np.asarray(latents).shape)
    arrays["latents" + suffix] = np.asarray(latents, dtype=np.float32)
    arrays["original_size" + suffix] = np.array(original_size)
    arrays["crop_ltrb" + suffix] = np.array(crop_ltrb)
    if flipped_latents is not None:
        arrays["latents_flipped" + suffix] = np.asarray(flipped_latents, dtype=np.float32)
    if alpha_mask is not None:
        arrays["alpha_mask" + suffix] = np.asarray(alpha_mask, dtype=np.float32)
    np.savez(path, **arrays)


def load_latents(path, bucket_reso, stride: int = LATENT_STRIDE) -> np.ndarray:
    height = bucket_reso[1] // stride
    width = bucket_reso[0] // stride
    suffix = f"_{height}x{width}"
    with np.load(path) as npz:
        if "latents" + suffix in npz.files:
            return npz["latents" + suffix].astype(np.float32)
        if "latents" in npz.files:
            return npz["latents"].astype(np.float32)
        for key in npz.files:
            if key.startswith("latents_"):
                return npz[key].astype(np.float32)
        raise KeyError(f"latents not found in {path} (tried latents{suffix}, latents and any latents_* key)")


def read_latents_original_size(path):
    with np.load(path) as npz:
        for key in npz.files:
            if key.startswith("original_size"):
                value = npz[key].tolist()
                if len(value) == 2:
                    return int(value[0]), int(value[1])
    return None


def save_te_outputs(path, candidates: List[Tuple[np.ndarray, str]], uncond_embeds, caption_dropout_rate: float = 0.0):
    arrays: Dict[str, np.ndarray] = {
        "num_candidates": np.array(len(candidates), dtype=np.int32),
        "uncond_embeds": float32_to_bf16_bits(uncond_embeds),
        "uncond_mask": np.ones(np.asarray(uncond_embeds).shape[0], dtype=np.int32),
        "caption_dropout_rate": np.array(caption_dropout_rate, dtype=np.float32),
    }
    for index, (embeds, caption) in enumerate(candidates):
        embeds = np.asarray(embeds)
        arrays[f"prompt_embeds_{index}"] = float32_to_bf16_bits(embeds)
        arrays[f"attn_mask_{index}"] = np.ones(embeds.shape[0], dtype=np.int32)
        arrays[f"caption_{index}"] = np.array(caption)
    np.savez(path, **arrays)


def load_te_outputs(path) -> Dict:
    with np.load(path) as npz:
        files = set(npz.files)
        num_candidates = int(npz["num_candidates"]) if "num_candidates" in files else 1
        candidates = []
        for index in range(num_candidates):
            if "num_candidates" in files:
                embeds_key = f"prompt_embeds_{index}"
                mask_key = f"attn_mask_{index}"
                caption_key = f"caption_{index}"
            else:
                embeds_key, mask_key, caption_key = "prompt_embeds", "attn_mask", None
            candidates.append(
                {
                    "embeds": uint16_to_float32(npz[embeds_key]),
                    "mask": npz[mask_key].astype(np.int32),
                    "caption": str(npz[caption_key]) if caption_key is not None and caption_key in files else "",
                }
            )
        uncond = None
        if "uncond_embeds" in files:
            uncond = {
                "embeds": uint16_to_float32(npz["uncond_embeds"]),
                "mask": npz["uncond_mask"].astype(np.int32),
            }
        dropout = float(npz["caption_dropout_rate"]) if "caption_dropout_rate" in files else 0.0
    return {"candidates": candidates, "uncond": uncond, "caption_dropout_rate": dropout}


def select_candidate(record: Dict, rng: np.random.Generator) -> Dict:
    index = int(rng.integers(0, len(record["candidates"])))
    return record["candidates"][index]


def pad_embeds(embeds_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    length = max(embeds.shape[0] for embeds in embeds_list)
    channels = embeds_list[0].shape[1]
    padded = np.zeros((len(embeds_list), length, channels), dtype=np.float32)
    mask = np.zeros((len(embeds_list), length), dtype=np.int32)
    for index, embeds in enumerate(embeds_list):
        padded[index, : embeds.shape[0]] = embeds
        mask[index, : embeds.shape[0]] = 1
    return padded, mask


def find_cache_pairs(directory: str, num_repeats: int = 1, caption_dropout_rate: Optional[float] = None) -> List[Dict]:
    records = []
    pattern = os.path.join(directory, "**", "*" + MOSS_TE_SUFFIX)
    for te_path in sorted(glob.glob(pattern, recursive=True)):
        stem = te_path[: -len(MOSS_TE_SUFFIX)]
        latents = sorted(glob.glob(glob.escape(stem) + "_[0-9]*x[0-9]*" + MOSS_LATENTS_SUFFIX))
        if not latents:
            latents = sorted(glob.glob(glob.escape(stem) + MOSS_LATENTS_SUFFIX))
        if latents:
            records.append(
                {
                    "key": stem,
                    "text_encoder": te_path,
                    "latents": latents,
                    "num_repeats": int(num_repeats),
                    "caption_dropout_rate": caption_dropout_rate,
                }
            )
    return records
