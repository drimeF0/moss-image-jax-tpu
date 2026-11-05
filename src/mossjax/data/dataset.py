from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from .. import vae as vae_mod
from . import cache as cache_mod

TEXT_LENGTH_BUCKETS = (128, 256, 512, 1024, 2048, 4096)


def pad_text_length(length: int, buckets: Sequence[int] = TEXT_LENGTH_BUCKETS) -> int:
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return buckets[-1]


def pad_to_length(embeds: np.ndarray, mask: np.ndarray, length: int):
    channels = embeds.shape[-1]
    padded_embeds = np.zeros((length, channels), dtype=np.float32)
    padded_mask = np.zeros(length, dtype=np.int32)
    valid = min(embeds.shape[0], length)
    padded_embeds[:valid] = embeds[:valid]
    padded_mask[:valid] = mask[:valid]
    return padded_embeds, padded_mask


class CachedDataset:
    def __init__(
        self,
        records: Sequence[dict],
        bucket_reso: Tuple[int, int],
        buckets: Sequence[int] = TEXT_LENGTH_BUCKETS,
        caption_dropout: bool = True,
        seed: int = 0,
        vae_stats: Optional[dict] = None,
    ):
        if not records:
            raise ValueError("records must not be empty")
        self.records = list(records)
        self.bucket_reso = bucket_reso
        self.buckets = tuple(buckets)
        self.caption_dropout = caption_dropout
        self.seed = seed
        self.vae_stats = vae_stats
        self.rng = np.random.default_rng(seed)
        weights = np.array([max(1, int(record.get("num_repeats", 1))) for record in self.records], dtype=np.float64)
        self.weights = weights / weights.sum()

    def __len__(self):
        return len(self.records)

    def _load_record(self, record: dict):
        text_record = cache_mod.load_te_outputs(record["text_encoder"])
        candidate = cache_mod.select_candidate(text_record, self.rng)
        dropout_rate = record.get("caption_dropout_rate")
        if dropout_rate is None:
            dropout_rate = text_record["caption_dropout_rate"]
        use_uncond = (
            self.caption_dropout
            and text_record["uncond"] is not None
            and dropout_rate > 0.0
            and self.rng.random() < dropout_rate
        )
        if use_uncond:
            embeds, mask = text_record["uncond"]["embeds"], text_record["uncond"]["mask"]
        else:
            embeds, mask = candidate["embeds"], candidate["mask"]
        target_length = pad_text_length(embeds.shape[0], self.buckets)
        embeds, mask = pad_to_length(embeds, mask, target_length)
        bucket_reso = record.get("bucket", self.bucket_reso)
        latents = cache_mod.load_latents(record["latents"][0], tuple(bucket_reso))
        if self.vae_stats is not None:
            latents = np.asarray(vae_mod.patchify(jnp.asarray(latents)[None]), dtype=np.float32)[0]
            latents = np.asarray(
                vae_mod.normalize(jnp.asarray(latents)[None], self.vae_stats["running_mean"], self.vae_stats["running_var"]),
                dtype=np.float32,
            )[0]
        return latents, embeds, mask

    def sample(self, index: Optional[int] = None):
        if index is None:
            index = int(self.rng.choice(len(self.records), p=self.weights))
        record = self.records[index % len(self.records)]
        latents, embeds, mask = self._load_record(record)
        return {
            "latents": latents[None],
            "context": embeds[None],
            "context_mask": mask[None],
            "key": record["key"],
        }
