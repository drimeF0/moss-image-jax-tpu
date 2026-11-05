#!/usr/bin/env python3
"""Single-process runner for Colab/Kaggle TPU: smoke test or the real pipeline.

All JAX work happens in this one process (one process per TPU per sandbox
lifetime on gVisor-based runtimes). Stages are isolated: an error is recorded
in summary.json and later stages are skipped.

    python scripts/colab_run.py --mode smoke --output-dir /content/out
    python scripts/colab_run.py --mode full --dit ... --vae ... --text-encoder ... \
        --tokenizer ... --images-dir ... --prompts prompts.json --output-dir /content/out --int8
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from mossjax import caching, init, lora as lora_mod, quant, sampling, smoke, text_encoder as te_mod
from mossjax import train as train_mod, trainer
from mossjax import vae as vae_mod
from mossjax import weights
from mossjax.config import MOSS_IMAGE_CONFIG
from mossjax.data import cache as cache_mod


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="training steps (smoke default 6, full default 200)")
    parser.add_argument("--sample-steps", type=int, default=None, help="denoising steps (smoke default 4, full default 50)")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"])
    parser.add_argument("--int8", action="store_true", help="int8 for the DiT and TE (host-side quantization)")
    parser.add_argument("--no-cache", action="store_true", help="skip cache building (use existing caches)")
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--dit", default=None)
    parser.add_argument("--vae", default=None)
    parser.add_argument("--text-encoder", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--te-config", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--sample-prompt", default="a red fox in the snow")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--timestep-sampling", default="truncated_logit_normal")
    parser.add_argument("--query-chunk", type=int, default=0, help="attention query chunk for sampling (0 = auto)")
    return parser.parse_args()


def free(*objects):
    for obj in objects:
        del obj
    gc.collect()


def resolve_shards(path):
    import glob

    if "*" in path:
        return sorted(glob.glob(path))
    if "-00001-of-" in os.path.basename(path):
        return sorted(glob.glob(os.path.join(os.path.dirname(path), "*.safetensors")))
    return [path]


def device_report():
    import jax

    try:
        devices = jax.devices()
    except Exception as error:
        return {"backend": "unavailable", "error": f"{type(error).__name__}: {error}"}
    return {
        "backend": jax.default_backend(),
        "count": len(devices),
        "devices": [str(device) for device in devices],
    }


def vae_stats_from(params):
    return {
        "running_mean": np.asarray(params["bn"]["running_mean"]),
        "running_var": np.asarray(params["bn"]["running_var"]),
    }


def make_prompt_embeds(te_params, te_cfg, tokenizer, prompt, negative="", max_length=caching.MAX_TOKEN_LENGTH):
    prompt_ids = caching.tokenize_prompt(tokenizer, prompt, max_length)
    negative_ids = caching.tokenize_prompt(tokenizer, negative, max_length)
    embeds = caching.encode_prompt_embeds(te_params, te_cfg, [prompt_ids, negative_ids])
    prompt_embeds = np.asarray(embeds[0])[None]
    negative_embeds = np.asarray(embeds[1])[None]
    return prompt_embeds, np.ones(prompt_embeds.shape[:2], np.int32), negative_embeds, np.ones(negative_embeds.shape[:2], np.int32)


def run_smoke(args, output_dir, summary):
    dit_params, te_params, vae_params = smoke.build_params(args.seed)
    size = smoke.SMOKE_IMAGE_SIZE
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for index in range(2):
        image = rng.uniform(-1.0, 1.0, (3, size, size)).astype(np.float32)
        caching.write_latents_cache(
            os.path.join(cache_dir, f"img{index}_{size:04d}x{size:04d}_moss.npz"),
            vae_params,
            smoke.SMOKE_VAE,
            image,
            original_size=[size, size],
        )
        caching.write_te_cache(
            os.path.join(cache_dir, f"img{index}_moss_te.npz"),
            te_params,
            smoke.SMOKE_TE,
            [rng.integers(0, smoke.SMOKE_TE.vocab_size, size=9)],
            [f"smoke caption {index}"],
            rng.integers(0, smoke.SMOKE_TE.vocab_size, size=3),
        )

    records = cache_mod.find_cache_pairs(cache_dir)
    steps = args.steps if args.steps is not None else 6
    sample_steps = args.sample_steps if args.sample_steps is not None else 4
    print(f"[stage] smoke: {len(records)} cached samples, {steps} training steps", flush=True)
    train_config = trainer.TrainConfig(
        rank=args.rank, learning_rate=args.learning_rate or 0.05, optimizer=args.optimizer, remat=True
    )
    loop_config = train_mod.LoopConfig(steps=steps, seed=args.seed, timestep_sampling="uniform", log_every=1)
    state, history = train_mod.train_loop(
        dit_params,
        smoke.SMOKE_DIT,
        records,
        train_config,
        loop_config,
        bucket_reso=(size, size),
        output_dir=os.path.join(output_dir, "checkpoints"),
        vae_stats=vae_stats_from(vae_params),
    )
    summary["training"] = {"steps": steps, "first_loss": history[0], "last_loss": history[-1]}

    print(f"[stage] smoke: sampling {sample_steps} steps", flush=True)
    latents = sampling.generate_latents(
        dit_params,
        smoke.SMOKE_DIT,
        np.zeros((1, 4, smoke.SMOKE_DIT.text_dim), np.float32),
        np.ones((1, 4), np.int32),
        height=size,
        width=size,
        steps=sample_steps,
        guidance_scale=1.0,
        seed=args.seed,
    )
    pixels = np.asarray(vae_mod.decode_from_dit(vae_params, latents, smoke.SMOKE_VAE), dtype=np.float32)[0]
    pixels = np.clip((pixels.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    from PIL import Image

    sample_path = os.path.join(output_dir, "smoke_sample.png")
    Image.fromarray(pixels).save(sample_path)
    summary["sample"] = sample_path


def run_full(args, output_dir, summary):
    if not args.dit or not args.vae or not args.text_encoder or not args.tokenizer:
        raise SystemExit("--dit, --vae, --text-encoder and --tokenizer are required in full mode")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    te_cfg_path = args.te_config or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "reference", "kohya_sd_scripts", "configs", "moss_image_qwen35_9b", "config.json",
    )
    te_cfg = te_mod.load_config(te_cfg_path)

    cache_dir = args.cache_dir or os.path.join(output_dir, "cache")
    need_cache = not args.no_cache
    need_sample = not args.skip_sample
    sample_steps = args.sample_steps if args.sample_steps is not None else 50

    print("[stage] vae: loading", flush=True)
    vae_started = time.time()
    vae_params, vae_cfg = vae_mod.load_vae_params(weights.load_state_dict_from_safetensors([args.vae]))
    print(f"[stage] vae: loaded in {time.time() - vae_started:.1f}s", flush=True)

    prompt_embeds = prompt_mask = negative_embeds = negative_mask = None
    if need_cache or need_sample:
        print("[stage] text encoder: loading (caches and/or sample prompt)", flush=True)
        te_started = time.time()
        te_params = te_mod.load_params(resolve_shards(args.text_encoder), te_cfg, quantize_int8=args.int8)
        print(f"[stage] text encoder: loaded in {time.time() - te_started:.1f}s", flush=True)
        try:
            if need_cache:
                if not args.images_dir or not args.prompts:
                    raise SystemExit("--images-dir and --prompts are required to build caches (or pass --no-cache)")
                with open(args.prompts, "r", encoding="utf-8") as handle:
                    prompts = json.load(handle)
                print(f"[stage] cache: {args.images_dir} -> {cache_dir}", flush=True)
                cache_resolution = None if args.dataset_config else (args.width, args.height)
                cached = caching.build_caches_from_images(
                    args.images_dir,
                    prompts,
                    vae_params,
                    vae_cfg,
                    te_params,
                    te_cfg,
                    tokenizer,
                    cache_dir,
                    max_samples=args.max_samples,
                    resolution=cache_resolution,
                )
                summary["cache"] = {"samples": len(cached), "dir": cache_dir}
            if need_sample:
                print("[stage] sample: encoding the prompt", flush=True)
                prompt_embeds, prompt_mask, negative_embeds, negative_mask = make_prompt_embeds(
                    te_params, te_cfg, tokenizer, args.sample_prompt
                )
        finally:
            del te_params
            free()

    if args.dataset_config:
        from mossjax.data import build_records, load_dataset_config

        records = build_records(load_dataset_config(args.dataset_config))
    else:
        records = cache_mod.find_cache_pairs(cache_dir)
    if not records:
        raise SystemExit(f"no cache pairs found in {cache_dir}")

    print(f"[stage] train: {len(records)} samples, loading DiT", flush=True)
    dit_started = time.time()
    dit_params = weights.load_params(resolve_shards(args.dit), MOSS_IMAGE_CONFIG, quantize_int8=args.int8)
    print(f"[stage] train: DiT loaded in {time.time() - dit_started:.1f}s", flush=True)
    steps = args.steps if args.steps is not None else 200
    print(f"[stage] train: {steps} steps", flush=True)
    train_config = trainer.TrainConfig(
        rank=args.rank,
        learning_rate=args.learning_rate or 1e-4,
        optimizer=args.optimizer,
        remat=True,
    )
    loop_config = train_mod.LoopConfig(
        steps=steps, seed=args.seed, timestep_sampling=args.timestep_sampling, save_every=100
    )
    state, history = train_mod.train_loop(
        dit_params,
        MOSS_IMAGE_CONFIG,
        records,
        train_config,
        loop_config,
        bucket_reso=(args.width, args.height),
        output_dir=os.path.join(output_dir, "checkpoints"),
        vae_stats=vae_stats_from(vae_params),
    )
    summary["training"] = {"steps": steps, "first_loss": history[0], "last_loss": history[-1]}

    if need_sample:
        print("[stage] sample: denoising (DiT kept from training)", flush=True)
        query_chunk = args.query_chunk if args.query_chunk > 0 else None
        if query_chunk is None and (args.width // 16) * (args.height // 16) > 2048:
            query_chunk = 1024
        latents = sampling.generate_latents(
            dit_params,
            MOSS_IMAGE_CONFIG,
            prompt_embeds,
            prompt_mask,
            negative_embeds,
            negative_mask,
            height=args.height,
            width=args.width,
            steps=sample_steps,
            guidance_scale=3.0,
            flow_shift=5.0,
            uncond_skip_layers=(22,),
            seed=args.seed,
            query_chunk_size=query_chunk,
        )
        del dit_params
        free()
        print("[stage] sample: decoding with VAE", flush=True)
        pixels = np.asarray(vae_mod.decode_from_dit(vae_params, latents, vae_cfg), dtype=np.float32)[0]
        pixels = np.clip((pixels.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        from PIL import Image

        sample_path = os.path.join(output_dir, "sample.png")
        Image.fromarray(pixels).save(sample_path)
        summary["sample"] = sample_path


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    summary = {"mode": args.mode, "device": device_report(), "stages": {}}
    started = time.time()
    for name, stage in (("smoke", run_smoke), ("full", run_full)):
        if args.mode != name:
            continue
        stage_started = time.time()
        try:
            stage(args, output_dir, summary)
            summary["stages"][name] = {"status": "ok", "seconds": round(time.time() - stage_started, 2)}
        except Exception as error:
            summary["stages"][name] = {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "trace": traceback.format_exc()[-3000:],
            }
            break
    summary["seconds"] = round(time.time() - started, 2)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "device"}, indent=2))
    if summary["stages"].get(args.mode, {}).get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
