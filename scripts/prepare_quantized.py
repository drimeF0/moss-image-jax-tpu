#!/usr/bin/env python3
"""Pre-quantize MOSS-Image checkpoints to int8 (weights + fp32 scales) once.

The output safetensors stores quantized layers as ``<key>`` (int8) plus
``<key>.scale`` (fp32 [out]) with a metadata marker, so loaders pick them up
directly and skip the (slow) quantization: training-time loading becomes a plain
safetensors read.

    python scripts/prepare_quantized.py \
        --dit  /content/models/moss-image-base/transformer/diffusion_pytorch_model-00001-of-00003.safetensors \
        --text-encoder /content/models/moss-image-base/text_encoder/model-00001-of-00004.safetensors \
        --output-dir /content/models/prepared

Then pass the prepared files to train.py / colab_run.py / sample.py as --dit and
--text-encoder (no --int8 needed).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mossjax import prepare, text_encoder as te_mod

from train import resolve_shards


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", default=None, help="first shard or single-file safetensors of the DiT")
    parser.add_argument("--text-encoder", default=None, help="first shard or single-file safetensors of the TE")
    parser.add_argument("--te-config", default=None, help="Qwen3.5-9B config.json (defaults to the vendored reference)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="moss_image_base", help="output file prefix")
    parser.add_argument("--skip-dit", action="store_true")
    parser.add_argument("--skip-te", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.dit and not args.text_encoder and not args.skip_dit and not args.skip_te:
        raise SystemExit("nothing to do: pass --dit and/or --text-encoder")
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}

    if args.dit and not args.skip_dit:
        output_path = os.path.join(args.output_dir, f"{args.name}_dit_int8.safetensors")
        summary["dit"] = prepare.prepare_dit(resolve_shards(args.dit), output_path)
        print(f"DiT: {summary['dit']}")

    if args.text_encoder and not args.skip_te:
        config_path = args.te_config
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "tests", "reference", "kohya_sd_scripts", "configs", "moss_image_qwen35_9b", "config.json",
            )
        te_cfg = te_mod.load_config(config_path)
        output_path = os.path.join(args.output_dir, f"{args.name}_te_int8.safetensors")
        summary["text_encoder"] = prepare.prepare_text_encoder(resolve_shards(args.text_encoder), te_cfg, output_path)
        print(f"TE: {summary['text_encoder']}")

    summary_path = os.path.join(args.output_dir, "prepare_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
