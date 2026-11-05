from typing import List, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from . import text_encoder as text_encoder_mod
from . import vae as vae_mod
from .data import cache as cache_mod


def encode_prompt_embeds(params, cfg, input_ids_list: Sequence[np.ndarray]) -> List[np.ndarray]:
    outputs = []
    for input_ids in input_ids_list:
        hidden = text_encoder_mod.forward(params, jnp.asarray(np.asarray(input_ids)), cfg)
        outputs.append(np.asarray(hidden[0], dtype=np.float32))
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
    latents = np.asarray(vae_mod.encode(params, x, cfg), dtype=np.float32)[0]
    height, width = latents.shape[-2:]
    original = list(original_size) if original_size is not None else [latents.shape[-1] * 8, latents.shape[-2] * 8]
    cache_mod.save_latents(
        path,
        latents,
        original_size=original,
        crop_ltrb=[0, 0, original[0], original[1]],
    )
    return latents
