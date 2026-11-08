#!/usr/bin/env python3
"""Build *_moss.npz (FLUX.2 VAE latents) and *_moss_te.npz (Qwen3.5 TE outputs) caches.

Example:
    python scripts/make_caches.py --images-dir data/images --prompts prompts.json \
        --vae flux2-vae.safetensors --text-encoder text_encoder/model-00001-of-00004.safetensors \
        --tokenizer tokenizer --output-dir data/cache --int8-te
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from PIL import Image

from mossjax import caching, quant, text_encoder as te_mod, vae as vae_mod
from mossjax import weights
from mossjax.config import MOSS_IMAGE_CONFIG
from mossjax.data import buckets

MAX_TOKEN_LENGTH = 4096


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--prompts", required=True, help="JSON: {image filename: [caption, ...]}")
    parser.add_argument("--vae", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--tokenizer", required=True, help="tokenizer directory")
    parser.add_argument("--te-config", default=None, help="Qwen3.5-9B config.json (defaults to the vendored reference)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-resolution", type=int, nargs=2, default=[1024, 1024])
    parser.add_argument("--resolution", type=int, nargs=2, default=None, help="fixed target W H (default: per-image bucketing)")
    parser.add_argument("--bucket-steps", type=int, default=64)
    parser.add_argument("--caption-dropout-rate", type=float, default=0.0)
    parser.add_argument("--int8-te", action="store_true")
    parser.add_argument("--force-cache", action="store_true", help="rebuild all caches even if valid ones exist")
    return parser.parse_args()


def format_prompt(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n thinking\n"


def tokenize(tokenizer, prompt, max_length=MAX_TOKEN_LENGTH):
    ids = tokenizer(format_prompt(prompt))["input_ids"]
    return np.asarray(ids[:max_length], dtype=np.int64)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    vae_params, vae_cfg = vae_mod.load_vae_params(weights.load_state_dict_from_safetensors([args.vae]))
    config_path = args.te_config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "reference", "kohya_sd_scripts", "configs", "moss_image_qwen35_9b", "config.json",
        )
    te_cfg = te_mod.load_config(config_path)
    te_params = te_mod.load_params([args.text_encoder], te_cfg, quantize_int8=args.int8_te)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.prompts, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)

    cached = caching.build_caches_from_images(
        args.images_dir,
        prompts,
        vae_params,
        vae_cfg,
        te_params,
        te_cfg,
        tokenizer,
        args.output_dir,
        max_resolution=tuple(args.max_resolution),
        bucket_steps=args.bucket_steps,
        caption_dropout_rate=args.caption_dropout_rate,
        resolution=tuple(args.resolution) if args.resolution else None,
        overwrite_existing=args.force_cache,
    )
    print(f"cached {len(cached)} images into {args.output_dir}")


if __name__ == "__main__":
    main()
