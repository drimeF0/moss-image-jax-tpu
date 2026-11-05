from typing import List, Optional, Sequence

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from . import text_encoder as text_encoder_mod
from . import vae as vae_mod
from .data import cache as cache_mod
from .data.dataset import TEXT_LENGTH_BUCKETS, pad_text_length

MAX_TOKEN_LENGTH = 4096


@lru_cache(maxsize=None)
def _te_forward_fn(cfg):
    return jax.jit(lambda params, input_ids: text_encoder_mod.forward(params, input_ids, cfg))


@lru_cache(maxsize=None)
def _vae_encode_fn(cfg):
    return jax.jit(lambda params, pixels: vae_mod.encode(params, pixels, cfg))


def format_prompt(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n thinking\n"


def tokenize_prompt(tokenizer, prompt: str, max_length: int = MAX_TOKEN_LENGTH) -> np.ndarray:
    ids = tokenizer(format_prompt(prompt))["input_ids"]
    return np.asarray(ids[:max_length], dtype=np.int64)


def encode_prompt_embeds(
    params,
    cfg,
    input_ids_list: Sequence[np.ndarray],
    jit: bool = True,
    buckets: Sequence[int] = TEXT_LENGTH_BUCKETS,
    pad_token_id: int = 0,
) -> List[np.ndarray]:
    outputs = []
    for input_ids in input_ids_list:
        ids = np.asarray(input_ids)
        valid_length = ids.shape[0]
        if jit:
            target_length = pad_text_length(valid_length, buckets) if valid_length <= buckets[-1] else valid_length
            padded = np.full(target_length, int(pad_token_id), dtype=np.int64)
            padded[:valid_length] = ids
            hidden = _te_forward_fn(cfg)(params, jnp.asarray(padded))
        else:
            hidden = text_encoder_mod.forward(params, jnp.asarray(ids), cfg)
        outputs.append(np.asarray(hidden[0, :valid_length], dtype=np.float32))
    return outputs


def write_te_cache(
    path,
    params,
    cfg,
    caption_ids_list: Sequence[np.ndarray],
    captions: Sequence[str],
    uncond_ids: np.ndarray,
    caption_dropout_rate: float = 0.0,
):
    if len(caption_ids_list) != len(captions):
        raise ValueError("caption_ids_list and captions must have the same length")
    embeds = encode_prompt_embeds(params, cfg, caption_ids_list)
    uncond = encode_prompt_embeds(params, cfg, [uncond_ids])[0]
    cache_mod.save_te_outputs(path, list(zip(embeds, captions)), uncond, caption_dropout_rate)


def write_latents_cache(path, params, cfg, pixels: np.ndarray, original_size: Optional[Sequence[int]] = None):
    x = jnp.asarray(np.asarray(pixels, dtype=np.float32))[None]
    latents = np.asarray(_vae_encode_fn(cfg)(params, x), dtype=np.float32)[0]
    height, width = latents.shape[-2:]
    original = list(original_size) if original_size is not None else [latents.shape[-1] * 8, latents.shape[-2] * 8]
    cache_mod.save_latents(
        path,
        latents,
        original_size=original,
        crop_ltrb=[0, 0, original[0], original[1]],
    )
    return latents


def build_caches_from_images(
    images_dir: str,
    prompts: dict,
    vae_params,
    vae_cfg,
    te_params,
    te_cfg,
    tokenizer,
    output_dir: str,
    max_resolution=(1024, 1024),
    bucket_steps: int = 64,
    caption_dropout_rate: float = 0.0,
    max_samples: Optional[int] = None,
    resolution: Optional[Sequence[int]] = None,
    progress: bool = True,
    log_every: int = 1,
):
    import os
    import time

    from PIL import Image

    from .data import buckets

    os.makedirs(output_dir, exist_ok=True)
    manager = None
    if resolution is None:
        manager = buckets.BucketManager(False, tuple(max_resolution), 256, max(max_resolution), bucket_steps)
        manager.make_buckets()
    uncond_ids = tokenize_prompt(tokenizer, "")
    items = list(prompts.items())
    if max_samples is not None:
        items = items[:max_samples]
    total_started = time.time()
    cached = []
    for index, (name, captions) in enumerate(items):
        image_started = time.time()
        image_path = os.path.join(images_dir, name)
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        if manager is not None:
            reso, _, _ = manager.select_bucket(*image.size)
            target_size = tuple(reso)
        else:
            target_size = tuple(resolution)
        image = image.resize(target_size, Image.LANCZOS)
        width, height = target_size
        pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        pixels = pixels.transpose(2, 0, 1)

        stem = os.path.splitext(os.path.join(output_dir, name))[0]
        parent = os.path.dirname(stem)
        if parent:
            os.makedirs(parent, exist_ok=True)
        latents_path = stem + f"_{width:04d}x{height:04d}_moss.npz"
        write_latents_cache(
            latents_path, vae_params, vae_cfg, pixels, original_size=[original_size[0], original_size[1]]
        )
        latents_done = time.time()

        te_path = stem + "_moss_te.npz"
        caption_ids = [tokenize_prompt(tokenizer, caption) for caption in captions]
        write_te_cache(
            te_path, te_params, te_cfg, caption_ids, captions, uncond_ids, caption_dropout_rate=caption_dropout_rate
        )
        te_done = time.time()
        cached.append(name)

        if progress and index == 0:
            print("[cache] first image includes XLA compilation for the current shapes", flush=True)
        if progress and ((index + 1) % max(log_every, 1) == 0 or index == len(items) - 1):
            elapsed = te_done - total_started
            eta = elapsed / (index + 1) * (len(items) - index - 1)
            print(
                f"[cache {index + 1}/{len(items)}] {name}: latents {latents_done - image_started:.1f}s, "
                f"te {te_done - latents_done:.1f}s, elapsed {elapsed:.0f}s, eta {eta:.0f}s",
                flush=True,
            )
    if progress and items:
        print(
            f"[cache] done: {len(cached)} images in {time.time() - total_started:.0f}s -> {output_dir}",
            flush=True,
        )
    return cached
