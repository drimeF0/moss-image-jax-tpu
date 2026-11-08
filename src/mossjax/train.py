import os
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from . import lora as lora_mod
from . import losses, trainer
from .config import MossConfig
from .data.dataset import CachedDataset


@dataclass
class LoopConfig:
    steps: int = 100
    seed: int = 0
    timestep_sampling: str = "truncated_logit_normal"
    discrete_flow_shift: float = 1.0
    sigmoid_scale: float = 1.0
    weighting_scheme: str = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    timestep_cutoff: float = 1.0
    timestep_uniform_ratio: float = 0.0
    log_every: int = 1
    save_every: int = 0
    checkpoint_metadata: dict = field(default_factory=dict)


def _flush_print(message):
    print(message, flush=True)


def device_memory_line():
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return ""
    if not stats or "bytes_in_use" not in stats:
        return ""
    used = stats["bytes_in_use"] / 1e9
    limit = stats.get("bytes_limit", 0) / 1e9
    peak = stats.get("peak_bytes_in_use", stats["bytes_in_use"]) / 1e9
    return f", mem {used:.1f}/{limit:.1f} GB (peak {peak:.1f})"


def train_loop(
    params,
    cfg: MossConfig,
    records,
    train_config: trainer.TrainConfig,
    loop_config: LoopConfig,
    bucket_reso,
    output_dir=None,
    vae_stats=None,
    log_fn=_flush_print,
    initial_lora=None,
):
    dataset = CachedDataset(records, bucket_reso, seed=loop_config.seed + 1, vae_stats=vae_stats)
    state, optimizer = trainer.init_train_state(
        params, cfg, train_config, jax.random.PRNGKey(loop_config.seed), initial_lora=initial_lora
    )
    step_fn = trainer.make_train_step(cfg, train_config, optimizer)
    rng = np.random.default_rng(loop_config.seed)
    checkpoint_dim = train_config.rank
    checkpoint_alpha = train_config.alpha if train_config.alpha is not None else float(train_config.rank)
    if initial_lora is not None:
        checkpoint_dim, checkpoint_alpha = lora_mod.lora_rank_alpha(initial_lora, checkpoint_dim, checkpoint_alpha)
        log_fn(f"[train] resuming from an existing LoRA: dim {checkpoint_dim}, alpha {checkpoint_alpha:g}")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    history = []
    seen_shapes = set()
    for step in range(1, loop_config.steps + 1):
        sample = dataset.sample()
        latents = sample["latents"]
        context = sample["context"]
        mask = sample["context_mask"]
        shape_key = (latents.shape, int(context.shape[1]))
        if shape_key not in seen_shapes:
            seen_shapes.add(shape_key)
            log_fn(
                f"[train] new input shape #{len(seen_shapes)}: latents {tuple(latents.shape)}, "
                f"text {context.shape[1]} (new compilation + device workspace)"
            )
        batch_size, _, height, width = latents.shape
        sigmas = losses.sample_sigmas(
            loop_config.timestep_sampling,
            batch_size,
            height,
            width,
            discrete_flow_shift=loop_config.discrete_flow_shift,
            sigmoid_scale=loop_config.sigmoid_scale,
            weighting_scheme=loop_config.weighting_scheme,
            logit_mean=loop_config.logit_mean,
            logit_std=loop_config.logit_std,
            mode_scale=loop_config.mode_scale,
            timestep_cutoff=loop_config.timestep_cutoff,
            timestep_uniform_ratio=loop_config.timestep_uniform_ratio,
            rng=rng,
        )
        noise = rng.standard_normal(latents.shape, dtype=np.float32)
        weighting = losses.compute_loss_weighting(loop_config.weighting_scheme, sigmas)
        batch = tuple(jnp.asarray(x) for x in (latents, noise, sigmas, context, mask, weighting))
        state, metrics = step_fn(state, params, *batch)
        loss = float(metrics["loss"])
        grad_norm = float(metrics["grad_norm"])
        history.append(loss)
        if loop_config.log_every and step % loop_config.log_every == 0:
            log_fn(
                f"steps: {step}, loss/current: {loss:.6f}, loss/average: {np.mean(history[-50:]):.6f}, "
                f"grad_norm: {grad_norm:.4f}{device_memory_line()}"
            )
        if output_dir is not None and loop_config.save_every and step % loop_config.save_every == 0:
            lora_mod.save_lora_file(
                os.path.join(output_dir, f"lora_step{step:07d}.safetensors"),
                state["lora"],
                cfg,
                dim=checkpoint_dim,
                alpha=checkpoint_alpha,
                metadata={"ss_steps": step, **loop_config.checkpoint_metadata},
            )

    if output_dir is not None:
        lora_mod.save_lora_file(
            os.path.join(output_dir, "moss_lora_final.safetensors"),
            state["lora"],
            cfg,
            dim=checkpoint_dim,
            alpha=checkpoint_alpha,
            metadata={"ss_steps": loop_config.steps, **loop_config.checkpoint_metadata},
        )
    return state, history
