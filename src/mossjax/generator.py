import gc
from typing import Callable, Optional

import numpy as np

from . import caching, sampling, vae as vae_mod
from .config import MossConfig

DEFAULT_UNCOND_SKIP_LAYERS = (22,)


def scale_lora(lora_tree, multiplier: float):
    if lora_tree is None or multiplier == 1.0:
        return lora_tree
    layers = []
    for block in lora_tree["layers"]:
        projections = {}
        for name, leaf in block["proj"].items():
            projections[name] = {**leaf, "scale": leaf["scale"] * multiplier}
        layers.append({"proj": projections})
    return {"layers": layers}


class MossImageGenerator:
    def __init__(
        self,
        dit_params,
        dit_cfg: MossConfig,
        te_params,
        te_cfg,
        vae_params,
        vae_cfg,
        tokenizer,
        lora=None,
        te_reload_fn: Optional[Callable] = None,
        free_te_after_encode: bool = False,
        query_chunk_size: Optional[int] = None,
    ):
        self.dit_params = dit_params
        self.dit_cfg = dit_cfg
        self.te_cfg = te_cfg
        self.vae_params = vae_params
        self.vae_cfg = vae_cfg
        self.tokenizer = tokenizer
        self.lora = lora
        self.te_reload_fn = te_reload_fn
        self.free_te_after_encode = free_te_after_encode
        self.query_chunk_size = query_chunk_size
        self._te_params = te_params

    def release_te(self):
        self._te_params = None
        gc.collect()

    def _loaded_te(self):
        if self._te_params is None:
            if self.te_reload_fn is None:
                raise RuntimeError("text encoder is not loaded and no reload function was provided")
            self._te_params = self.te_reload_fn()
        return self._te_params

    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        te_params = self._loaded_te()
        prompt_ids = caching.tokenize_prompt(self.tokenizer, prompt)
        negative_ids = caching.tokenize_prompt(self.tokenizer, negative_prompt)
        embeds = caching.encode_prompt_embeds(te_params, self.te_cfg, [prompt_ids, negative_ids])
        prompt_embeds = np.asarray(embeds[0])[None]
        negative_embeds = np.asarray(embeds[1])[None]
        if self.free_te_after_encode:
            self.release_te()
        return (
            prompt_embeds,
            np.ones(prompt_embeds.shape[:2], np.int32),
            negative_embeds,
            np.ones(negative_embeds.shape[:2], np.int32),
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 50,
        guidance_scale: float = 3.0,
        flow_shift: float = 5.0,
        seed: Optional[int] = None,
        lora_multiplier: float = 1.0,
        uncond_skip_layers=DEFAULT_UNCOND_SKIP_LAYERS,
        progress: bool = False,
    ) -> np.ndarray:
        prompt_embeds, prompt_mask, negative_embeds, negative_mask = self.encode_prompt(prompt, negative_prompt)
        query_chunk = self.query_chunk_size
        if query_chunk is None and (width // 16) * (height // 16) > 2048:
            query_chunk = 1024
        latents = sampling.generate_latents(
            self.dit_params,
            self.dit_cfg,
            prompt_embeds,
            prompt_mask,
            negative_embeds,
            negative_mask,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            uncond_skip_layers=tuple(uncond_skip_layers),
            seed=seed,
            lora=scale_lora(self.lora, lora_multiplier),
            query_chunk_size=query_chunk,
            progress=progress,
        )
        pixels = np.asarray(vae_mod.decode_from_dit(self.vae_params, latents, self.vae_cfg), dtype=np.float32)[0]
        return np.clip((pixels.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
