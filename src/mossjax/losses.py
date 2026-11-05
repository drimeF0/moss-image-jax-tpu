import math
from statistics import NormalDist
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

NUM_TRAIN_TIMESTEPS = 1000


def flow_scheduler(num_train_timesteps: int = NUM_TRAIN_TIMESTEPS, shift: float = 1.0):
    timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
    sigmas = timesteps / num_train_timesteps
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    return (sigmas * num_train_timesteps).astype(np.float32), sigmas.astype(np.float32)


def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _ndtr(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def sample_truncated_logit_normal_indices(
    batch_size: int,
    num_timesteps: int,
    logit_mean: float,
    logit_std: float,
    cutoff: float,
    uniform_ratio: float,
    logit_offset: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    uniforms: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    if not (0.0 < cutoff <= 1.0):
        raise ValueError(f"cutoff must be in (0, 1], got {cutoff}")
    if not (0.0 <= uniform_ratio <= 1.0):
        raise ValueError(f"uniform_ratio must be in [0, 1], got {uniform_ratio}")

    if uniforms is None:
        if rng is None:
            rng = np.random.default_rng()
        p_uniform = rng.random(batch_size, dtype=np.float32)
        if uniform_ratio > 0.0:
            use_uniform = rng.random(batch_size, dtype=np.float32)
            uniform_indices = rng.integers(0, num_timesteps, size=batch_size)
        else:
            use_uniform = np.zeros(batch_size, dtype=np.float32)
            uniform_indices = np.zeros(batch_size, dtype=np.int64)
    else:
        p_uniform, use_uniform, uniform_indices = uniforms

    mean = np.full(batch_size, float(logit_mean), dtype=np.float64)
    if logit_offset is not None:
        mean = mean + np.asarray(logit_offset, dtype=np.float64)
    std = float(logit_std)

    if cutoff < 1.0:
        z_lo = -math.log(cutoff / (1.0 - cutoff))
        p_lo = _ndtr((z_lo - mean) / std)
    else:
        p_lo = np.zeros_like(mean)
    p = p_lo + p_uniform.astype(np.float64) * (1.0 - p_lo)
    p = np.clip(p, 0.0, 1.0 - 1e-12)

    normal = NormalDist()
    ndtri = np.vectorize(lambda value: normal.inv_cdf(value) if value > 0.0 else -37.0)
    z = mean + std * ndtri(p)

    s = _sigmoid(-z) / cutoff
    indices = (num_timesteps - np.ceil(s * num_timesteps).astype(np.int64)).clip(0, num_timesteps - 1)

    if uniform_ratio > 0.0:
        use = use_uniform < uniform_ratio
        indices = np.where(use, np.asarray(uniform_indices, dtype=np.int64), indices)
    return indices


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    logit_offset: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    uniforms: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    if uniforms is None:
        if rng is None:
            rng = np.random.default_rng()
        if weighting_scheme == "logit_normal":
            normal_sample = rng.standard_normal(batch_size, dtype=np.float32)
            rand_sample = np.zeros(batch_size, dtype=np.float32)
        else:
            normal_sample = np.zeros(batch_size, dtype=np.float32)
            rand_sample = rng.random(batch_size, dtype=np.float32)
    else:
        normal_sample, rand_sample = uniforms

    if weighting_scheme == "logit_normal":
        u = normal_sample.astype(np.float64)
        if logit_offset is not None:
            u = u + np.asarray(logit_offset, dtype=np.float64)
        u = 1.0 / (1.0 + np.exp(-u))
    elif weighting_scheme == "mode":
        u = rand_sample.astype(np.float64)
        u = 1.0 - u - mode_scale * (np.cos(math.pi * u / 2.0) ** 2 - 1.0 + u)
    else:
        u = rand_sample.astype(np.float64)
    return u


def time_shift(mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def get_lin_function(x1: float = 256.0, y1: float = 0.5, x2: float = 4096.0, y2: float = 1.15):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return lambda x: slope * x + intercept


def sample_sigmas(
    mode: str,
    batch_size: int,
    latent_height: int,
    latent_width: int,
    discrete_flow_shift: float = 1.0,
    sigmoid_scale: float = 1.0,
    weighting_scheme: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    timestep_cutoff: float = 1.0,
    timestep_uniform_ratio: float = 0.0,
    timestep_sampling_offset: Optional[np.ndarray] = None,
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
    rng: Optional[np.random.Generator] = None,
):
    if rng is None:
        rng = np.random.default_rng()
    offset = None if timestep_sampling_offset is None else np.asarray(timestep_sampling_offset, dtype=np.float32)

    if mode == "uniform":
        sigmas = rng.random(batch_size, dtype=np.float32)
    elif mode == "sigmoid":
        r = rng.standard_normal(batch_size, dtype=np.float32)
        if offset is not None:
            r = r + offset
        sigmas = (1.0 / (1.0 + np.exp(-(sigmoid_scale * r)))).astype(np.float32)
    elif mode == "shift":
        r = rng.standard_normal(batch_size, dtype=np.float32)
        if offset is not None:
            r = r + offset
        s = 1.0 / (1.0 + np.exp(-(sigmoid_scale * r)))
        sigmas = (s * discrete_flow_shift / (1.0 + (discrete_flow_shift - 1.0) * s)).astype(np.float32)
    elif mode == "flux_shift":
        r = rng.standard_normal(batch_size, dtype=np.float32)
        if offset is not None:
            r = r + offset
        s = 1.0 / (1.0 + np.exp(-(sigmoid_scale * r)))
        mu = get_lin_function(y1=0.5, y2=1.15)((latent_height // 2) * (latent_width // 2))
        sigmas = time_shift(mu, 1.0, s).astype(np.float32)
    elif mode == "truncated_logit_normal":
        logit_offset = None if offset is None else -offset
        indices = sample_truncated_logit_normal_indices(
            batch_size, num_train_timesteps, logit_mean, logit_std, timestep_cutoff, timestep_uniform_ratio, logit_offset, rng
        )
        grid_timesteps, grid_sigmas = flow_scheduler(num_train_timesteps, discrete_flow_shift)
        sigmas = grid_sigmas[indices]
    else:
        logit_offset = None
        if offset is not None and weighting_scheme == "logit_normal":
            logit_offset = -offset.astype(np.float64)
        if weighting_scheme == "logit_normal":
            normal_draw = rng.standard_normal(batch_size, dtype=np.float32)
            uniform_draw = np.zeros(batch_size, dtype=np.float32)
        else:
            uniform_draw = rng.random(batch_size, dtype=np.float32)
            normal_draw = np.zeros(batch_size, dtype=np.float32)
        u = compute_density_for_timestep_sampling(
            weighting_scheme,
            batch_size,
            logit_mean,
            logit_std,
            mode_scale,
            logit_offset,
            uniforms=(normal_draw, uniform_draw),
        )
        indices = (u * num_train_timesteps).astype(np.int64)
        indices = np.clip(indices, 0, num_train_timesteps - 1)
        grid_timesteps, grid_sigmas = flow_scheduler(num_train_timesteps, discrete_flow_shift)
        sigmas = grid_sigmas[indices]

    return sigmas.astype(np.float32)


def compute_loss_weighting(weighting_scheme: str, sigmas):
    sigmas = np.asarray(sigmas, dtype=np.float32)
    if weighting_scheme == "sigma_sqrt":
        return (sigmas**-2.0).astype(np.float32)
    if weighting_scheme == "cosmap":
        bot = 1.0 - 2.0 * sigmas + 2.0 * sigmas**2
        return (2.0 / (math.pi * bot)).astype(np.float32)
    return np.ones_like(sigmas)


def noisy_model_input(latents, noise, sigmas):
    sigmas = jnp.asarray(sigmas).reshape((-1,) + (1,) * (latents.ndim - 1))
    return (1.0 - sigmas) * latents + sigmas * noise


def rectified_flow_loss(prediction, latents, noise, sigmas, weighting=None, loss_type="mse", huber_c=0.1, loss_weights=None):
    prediction = prediction.astype(jnp.float32)
    target = (noise - latents).astype(jnp.float32)
    if loss_type == "mse":
        loss = jnp.square(prediction - target)
    elif loss_type == "huber":
        delta = jnp.abs(prediction - target)
        loss = jnp.where(delta <= huber_c, 0.5 * delta**2, huber_c * (delta - 0.5 * huber_c))
    else:
        raise ValueError(f"unknown loss type {loss_type}")
    if weighting is not None:
        weighting = jnp.asarray(weighting).reshape((-1,) + (1,) * (loss.ndim - 1))
        loss = loss * weighting
    loss = jnp.mean(loss, axis=tuple(range(1, loss.ndim)))
    if loss_weights is not None:
        loss = loss * jnp.asarray(loss_weights)
    return jnp.mean(loss)
