from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np

from . import model as model_mod
from .losses import NUM_TRAIN_TIMESTEPS


def _sigma_grid(count: int, shift: float, total: int) -> np.ndarray:
    lower = shift / (total + shift - 1)
    values = np.linspace(1.0, lower, count, dtype=np.float32)
    return shift * values / (1.0 + (shift - 1.0) * values)


def sampling_sigmas(num_inference_steps: int, shift: float = 5.0, training_shift: float = 1.0, num_train_timesteps: int = NUM_TRAIN_TIMESTEPS) -> np.ndarray:
    if num_inference_steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("steps and shifts must be positive")
    total = num_train_timesteps
    sigmas = _sigma_grid(num_inference_steps, shift, total)
    training = _sigma_grid(total, training_shift, total)[::-1].copy()
    upper = np.searchsorted(training, sigmas).clip(max=total - 1)
    lower = (upper - 1).clip(min=0)
    nearest = np.where(np.abs(sigmas - training[lower]) <= np.abs(training[upper] - sigmas), lower, upper)
    snapped = training[nearest]
    unique = snapped[np.concatenate(([True], np.diff(snapped) != 0))]
    return np.concatenate((unique, np.zeros(1, dtype=np.float32))).astype(np.float32)


def euler_step(latents, model_pred, sigmas, step_index):
    dt = sigmas[step_index + 1] - sigmas[step_index]
    return latents + dt * model_pred


def generate_latents(
    params,
    cfg,
    prompt_embeds,
    prompt_mask,
    negative_embeds: Optional[np.ndarray] = None,
    negative_mask: Optional[np.ndarray] = None,
    height: int = 1024,
    width: int = 1024,
    steps: int = 50,
    guidance_scale: float = 3.0,
    flow_shift: float = 5.0,
    uncond_skip_layers: Sequence[int] = (),
    seed: Optional[int] = None,
    initial_latents: Optional[np.ndarray] = None,
    lora=None,
    remat: bool = False,
    rng: Optional[np.random.Generator] = None,
):
    height = max(16, height - height % 16)
    width = max(16, width - width % 16)
    shape = (1, cfg.latent_channels, height // 16, width // 16)
    if initial_latents is not None:
        latents = np.asarray(initial_latents, dtype=np.float32)
    else:
        if rng is None:
            rng = np.random.default_rng(seed)
        latents = rng.standard_normal(shape, dtype=np.float32)

    prompt_embeds = jnp.asarray(prompt_embeds, dtype=jnp.float32)
    prompt_mask = jnp.asarray(prompt_mask, dtype=jnp.int32)
    do_cfg = guidance_scale != 1.0 and negative_embeds is not None
    if do_cfg:
        negative_embeds = jnp.asarray(negative_embeds, dtype=jnp.float32)
        negative_mask = jnp.asarray(negative_mask, dtype=jnp.int32)

    sigmas = sampling_sigmas(steps, flow_shift)
    for index in range(len(sigmas) - 1):
        timestep = jnp.asarray([sigmas[index] * NUM_TRAIN_TIMESTEPS], dtype=jnp.float32)
        x = jnp.asarray(latents, dtype=jnp.float32)
        prediction = np.asarray(
            model_mod.forward(params, x, timestep, prompt_embeds, prompt_mask, cfg, lora=lora, remat=remat),
            dtype=np.float32,
        )
        if do_cfg:
            uncond = np.asarray(
                model_mod.forward(
                    params, x, timestep, negative_embeds, negative_mask, cfg, skip_layers=uncond_skip_layers, lora=lora, remat=remat
                ),
                dtype=np.float32,
            )
            prediction = uncond + guidance_scale * (prediction - uncond)
        latents = euler_step(latents, prediction, sigmas, index)
    return latents
