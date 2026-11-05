#!/usr/bin/env python3
"""Train a MOSS-Image LoRA on precomputed caches (JAX/TPU or CPU).

Example:
    python scripts/train.py --dit /path/to/diffusion_pytorch_model-00001-of-00003.safetensors \
        --vae /path/to/flux2-vae.safetensors --cache-dir /path/to/cache \
        --output-dir /path/to/out --int8 --rank 16 --steps 1000
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from mossjax import quant, train as train_mod, trainer
from mossjax import vae as vae_mod
from mossjax import weights
from mossjax.config import MOSS_IMAGE_CONFIG
from mossjax.data import cache as cache_mod


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", required=True, help="first shard or single-file safetensors of the MOSS-Image DiT")
    parser.add_argument("--vae", default=None, help="FLUX.2 VAE safetensors (needed for latent normalization stats)")
    parser.add_argument("--cache-dir", default=None, help="directory with *_moss.npz and *_moss_te.npz caches")
    parser.add_argument("--dataset-config", default=None, help="sd-scripts style .toml dataset config (alternative to --cache-dir)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket-width", type=int, default=1024)
    parser.add_argument("--bucket-height", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"])
    parser.add_argument("--adafactor-decay-rate", type=float, default=0.8)
    parser.add_argument("--adafactor-clipping-threshold", type=float, default=1.0)
    parser.add_argument("--adafactor-momentum", type=float, default=None)
    parser.add_argument("--adafactor-relative-step", action="store_true")
    parser.add_argument("--adafactor-no-scale-parameter", action="store_true")
    parser.add_argument("--adafactor-eps", type=float, default=1e-30)
    parser.add_argument("--adafactor-min-dim-size-to-factor", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timestep-sampling", default="truncated_logit_normal")
    parser.add_argument("--discrete-flow-shift", type=float, default=1.0)
    parser.add_argument("--weighting-scheme", default="logit_normal")
    parser.add_argument("--logit-mean", type=float, default=0.0)
    parser.add_argument("--logit-std", type=float, default=1.0)
    parser.add_argument("--mode-scale", type=float, default=1.29)
    parser.add_argument("--sigmoid-scale", type=float, default=1.0)
    parser.add_argument("--timestep-cutoff", type=float, default=1.0)
    parser.add_argument("--timestep-uniform-ratio", type=float, default=0.0)
    parser.add_argument("--int8", action="store_true", help="quantize the frozen DiT base to int8")
    parser.add_argument("--no-remat", action="store_true", help="disable per-block gradient checkpointing")
    parser.add_argument("--loss-type", default="mse", choices=["mse", "huber"])
    return parser.parse_args()


def resolve_shards(path):
    if "*" in path:
        return sorted(glob.glob(path))
    if "-00001-of-" in os.path.basename(path):
        directory = os.path.dirname(path)
        return sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    return [path]


def main():
    args = parse_args()
    params = weights.load_params(resolve_shards(args.dit), MOSS_IMAGE_CONFIG)
    if args.int8:
        params = quant.quantize_params(params, MOSS_IMAGE_CONFIG)

    vae_stats = None
    if args.vae is not None:
        vae_params, _ = vae_mod.load_vae_params(weights.load_state_dict_from_safetensors([args.vae]))
        vae_stats = {
            "running_mean": np.asarray(vae_params["bn"]["running_mean"]),
            "running_var": np.asarray(vae_params["bn"]["running_var"]),
        }

    if args.cache_dir is None and args.dataset_config is None:
        raise SystemExit("either --cache-dir or --dataset-config is required")
    if args.dataset_config is not None:
        from mossjax.data import build_records, load_dataset_config

        dataset_config = load_dataset_config(args.dataset_config)
        records = build_records(dataset_config)
    else:
        records = cache_mod.find_cache_pairs(args.cache_dir)
    if not records:
        raise SystemExit(f"no cache pairs found in {args.cache_dir or args.dataset_config}")
    print(f"found {len(records)} cached samples")

    train_config = trainer.TrainConfig(
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        remat=not args.no_remat,
        loss_type=args.loss_type,
        optimizer=args.optimizer,
        adafactor_scale_parameter=not args.adafactor_no_scale_parameter,
        adafactor_relative_step=args.adafactor_relative_step,
        adafactor_decay_rate=args.adafactor_decay_rate,
        adafactor_clipping_threshold=args.adafactor_clipping_threshold,
        adafactor_momentum=args.adafactor_momentum,
        adafactor_eps=args.adafactor_eps,
        adafactor_min_dim_size_to_factor=args.adafactor_min_dim_size_to_factor,
    )
    loop_config = train_mod.LoopConfig(
        steps=args.steps,
        seed=args.seed,
        timestep_sampling=args.timestep_sampling,
        discrete_flow_shift=args.discrete_flow_shift,
        sigmoid_scale=args.sigmoid_scale,
        weighting_scheme=args.weighting_scheme,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
        timestep_cutoff=args.timestep_cutoff,
        timestep_uniform_ratio=args.timestep_uniform_ratio,
        save_every=args.save_every,
    )
    train_mod.train_loop(
        params,
        MOSS_IMAGE_CONFIG,
        records,
        train_config,
        loop_config,
        bucket_reso=(args.bucket_width, args.bucket_height),
        output_dir=args.output_dir,
        vae_stats=vae_stats,
    )


if __name__ == "__main__":
    main()
