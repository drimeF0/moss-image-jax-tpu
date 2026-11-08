#!/usr/bin/env python3
"""Interactive MOSS-Image generation via Gradio.

Keeps the DiT (and optionally the text encoder) resident between prompts:

    python scripts/gradio_app.py \
        --dit /content/models/prepared/moss_image_base_dit_int8.safetensors \
        --vae /content/models/moss-image-base/vae/diffusion_pytorch_model.safetensors \
        --text-encoder /content/models/prepared/moss_image_base_te_int8.safetensors \
        --tokenizer /content/models/moss-image-base/tokenizer \
        --lora /content/moss_out/run1/checkpoints/moss_lora_final.safetensors --share

On 16 GB HBM the DiT (~5 GB) and TE (~9.5 GB) both fit for <=768px. For 1024px
or if you hit RESOURCE_EXHAUSTED, pass --low-mem: the TE is then reloaded for
each prompt and freed before denoising.
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from mossjax import generator as generator_mod, lora as lora_mod, text_encoder as te_mod, vae as vae_mod
from mossjax import weights
from mossjax.config import MOSS_IMAGE_CONFIG

from train import resolve_shards


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--te-config", default=None)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--lora-multiplier", type=float, default=1.0)
    parser.add_argument("--int8", action="store_true", help="quantize on load (not needed for prepared files)")
    parser.add_argument("--low-mem", action="store_true", help="reload TE per prompt and free it before denoising")
    parser.add_argument("--query-chunk", type=int, default=0, help="attention query chunk (0 = auto)")
    parser.add_argument("--width", type=int, default=768, help="default width in the UI")
    parser.add_argument("--height", type=int, default=768, help="default height in the UI")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--uncond-skip-layers", default="22")
    parser.add_argument("--share", action="store_true", help="create a public gradio link (Colab)")
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def build_generator(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config_path = args.te_config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests", "reference", "kohya_sd_scripts", "configs", "moss_image_qwen35_9b", "config.json",
        )
    te_cfg = te_mod.load_config(config_path)

    print("[app] loading VAE", flush=True)
    vae_params, vae_cfg = vae_mod.load_vae_params(weights.load_state_dict_from_safetensors([args.vae]))

    te_paths = resolve_shards(args.text_encoder)

    def te_reload_fn():
        print("[app] loading text encoder", flush=True)
        started = time.time()
        params = te_mod.load_params(te_paths, te_cfg, quantize_int8=args.int8)
        print(f"[app] text encoder loaded in {time.time() - started:.1f}s", flush=True)
        return params

    te_params = None
    if not args.low_mem:
        te_params = te_reload_fn()

    print("[app] loading DiT", flush=True)
    started = time.time()
    dit_params = weights.load_params(resolve_shards(args.dit), MOSS_IMAGE_CONFIG, quantize_int8=args.int8)
    print(f"[app] DiT loaded in {time.time() - started:.1f}s", flush=True)

    lora_tree = None
    if args.lora:
        lora_tree = lora_mod.load_lora_file(args.lora, dit_params, MOSS_IMAGE_CONFIG)
        print("[app] LoRA loaded", flush=True)

    return generator_mod.MossImageGenerator(
        dit_params,
        MOSS_IMAGE_CONFIG,
        te_params,
        te_cfg,
        vae_params,
        vae_cfg,
        tokenizer,
        lora=lora_tree,
        te_reload_fn=te_reload_fn,
        free_te_after_encode=args.low_mem,
        query_chunk_size=args.query_chunk if args.query_chunk > 0 else None,
    )


def main():
    args = parse_args()
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit('gradio is not installed: pip install "gradio>=4"')

    generator = build_generator(args)
    skip_layers = tuple(int(value) for value in args.uncond_skip_layers.split(",") if value.strip())

    def run(prompt, negative_prompt, width, height, steps, guidance_scale, flow_shift, seed, lora_multiplier):
        if not prompt.strip():
            return None, "empty prompt"
        width = int(width - width % 16)
        height = int(height - height % 16)
        resolved_seed = int(seed) if seed is not None and int(seed) >= 0 else int(np.random.default_rng().integers(0, 2**31))
        started = time.time()
        print(
            f"[app] generating {width}x{height}, steps={int(steps)}, guidance={guidance_scale}, "
            f"shift={flow_shift}, seed={resolved_seed}",
            flush=True,
        )
        image = generator.generate(
            prompt,
            negative_prompt,
            width=width,
            height=height,
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            flow_shift=float(flow_shift),
            seed=resolved_seed,
            lora_multiplier=float(lora_multiplier),
            uncond_skip_layers=skip_layers,
            progress=True,
        )
        gc.collect()
        return image, f"{width}x{height}, {int(steps)} steps, seed {resolved_seed}, {time.time() - started:.1f}s"

    with gr.Blocks(title="MOSS-Image JAX/TPU") as app:
        gr.Markdown("# MOSS-Image (JAX/TPU)\nLoRA inference on TPU; first generation includes XLA compilation.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=3, placeholder="a red fox in the snow")
                negative_prompt = gr.Textbox(label="Negative prompt", lines=2, value="")
                with gr.Row():
                    width = gr.Slider(256, 1024, value=args.width, step=16, label="Width")
                    height = gr.Slider(256, 1024, value=args.height, step=16, label="Height")
                with gr.Row():
                    steps = gr.Slider(5, 100, value=args.steps, step=1, label="Steps")
                    guidance_scale = gr.Slider(1.0, 10.0, value=args.guidance_scale, step=0.1, label="CFG scale")
                with gr.Row():
                    flow_shift = gr.Slider(1.0, 10.0, value=args.flow_shift, step=0.1, label="Flow shift")
                    seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                lora_multiplier = gr.Slider(0.0, 2.0, value=args.lora_multiplier, step=0.05, label="LoRA weight")
                button = gr.Button("Generate", variant="primary")
            with gr.Column():
                output = gr.Image(label="Result", type="numpy", height=640)
                status = gr.Markdown("")
        button.click(
            run,
            inputs=[prompt, negative_prompt, width, height, steps, guidance_scale, flow_shift, seed, lora_multiplier],
            outputs=[output, status],
        )
        prompt.submit(
            run,
            inputs=[prompt, negative_prompt, width, height, steps, guidance_scale, flow_shift, seed, lora_multiplier],
            outputs=[output, status],
        )

    app.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
