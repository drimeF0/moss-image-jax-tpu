#!/usr/bin/env python3
"""Sample images with a MOSS-Image DiT (optionally merged with a trained LoRA).

Example:
    python scripts/sample.py --dit diffusion_pytorch_model-00001-of-00003.safetensors \
        --vae flux2-vae.safetensors --text-encoder model-00001-of-00004.safetensors \
        --tokenizer tokenizer --prompt "a red fox in the snow" --output out.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from PIL import Image

from mossjax import caching, lora as lora_mod, sampling, text_encoder as te_mod, vae as vae_mod
from mossjax import weights
from mossjax.config import MOSS_IMAGE_CONFIG

from train import resolve_shards  # noqa: E402  (same directory)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--te-config", default=None)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", default="sample.png")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--uncond-skip-layers", default="22")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--lora-multiplier", type=float, default=1.0)
    return parser.parse_args()


def format_prompt(prompt):
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n thinking\n"


def main():
    args = parse_args()
    params = weights.load_params(resolve_shards(args.dit), MOSS_IMAGE_CONFIG, quantize_int8=args.int8)
    lora_tree = None
    if args.lora:
        lora_tree = lora_mod.load_lora_file(args.lora, params, MOSS_IMAGE_CONFIG)

    config_path = args.te_config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "reference", "kohya_sd_scripts", "configs", "moss_image_qwen35_9b", "config.json",
        )
    te_cfg = te_mod.load_config(config_path)
    te_params = te_mod.load_params(resolve_shards(args.text_encoder), te_cfg, quantize_int8=args.int8)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt_ids = np.asarray(tokenizer(format_prompt(args.prompt))["input_ids"], dtype=np.int64)
    negative_ids = np.asarray(tokenizer(format_prompt(args.negative_prompt))["input_ids"], dtype=np.int64)
    prompt_embeds = caching.encode_prompt_embeds(te_params, te_cfg, [prompt_ids])[0][None]
    negative_embeds = caching.encode_prompt_embeds(te_params, te_cfg, [negative_ids])[0][None]
    prompt_mask = np.ones(prompt_embeds.shape[:2], dtype=np.int32)
    negative_mask = np.ones(negative_embeds.shape[:2], dtype=np.int32)

    skip_layers = tuple(int(v) for v in args.uncond_skip_layers.split(",") if v.strip())
    latents = sampling.generate_latents(
        params,
        MOSS_IMAGE_CONFIG,
        prompt_embeds,
        prompt_mask,
        negative_embeds,
        negative_mask,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        flow_shift=args.flow_shift,
        uncond_skip_layers=skip_layers,
        seed=args.seed,
        lora=lora_tree,
    )
    vae_params, vae_cfg = vae_mod.load_vae_params(weights.load_state_dict_from_safetensors([args.vae]))
    pixels = np.asarray(vae_mod.decode_from_dit(vae_params, latents, vae_cfg), dtype=np.float32)[0]
    pixels = np.clip((pixels.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
