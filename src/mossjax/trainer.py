from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import optax

from . import lora as lora_mod
from . import model as model_mod
from .config import MossConfig
from .losses import NUM_TRAIN_TIMESTEPS, noisy_model_input, rectified_flow_loss


@dataclass
class TrainConfig:
    rank: int = 4
    alpha: Optional[float] = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    remat: bool = True
    loss_type: str = "mse"
    huber_c: float = 0.1
    optimizer: str = "adamw"
    adafactor_scale_parameter: bool = True
    adafactor_relative_step: bool = False
    adafactor_decay_rate: float = 0.8
    adafactor_clipping_threshold: float = 1.0
    adafactor_momentum: Optional[float] = None
    adafactor_eps: float = 1e-30
    adafactor_min_dim_size_to_factor: int = 128
    query_chunk_size: Optional[int] = 1024


def make_optimizer(train_config: TrainConfig):
    if train_config.optimizer == "adamw":
        inner = optax.adamw(
            learning_rate=train_config.learning_rate,
            b1=train_config.beta1,
            b2=train_config.beta2,
            eps=train_config.eps,
            weight_decay=train_config.weight_decay,
        )
    elif train_config.optimizer == "adafactor":
        learning_rate = None if train_config.adafactor_relative_step else train_config.learning_rate
        inner = optax.adafactor(
            learning_rate=learning_rate,
            multiply_by_parameter_scale=train_config.adafactor_scale_parameter,
            decay_rate=train_config.adafactor_decay_rate,
            clipping_threshold=train_config.adafactor_clipping_threshold,
            momentum=train_config.adafactor_momentum,
            weight_decay_rate=train_config.weight_decay if train_config.weight_decay > 0 else None,
            eps=train_config.adafactor_eps,
            min_dim_size_to_factor=train_config.adafactor_min_dim_size_to_factor,
        )
    else:
        raise ValueError(f"unknown optimizer {train_config.optimizer!r}")
    return optax.chain(optax.clip_by_global_norm(train_config.max_grad_norm), inner)


def init_train_state(params, cfg: MossConfig, train_config: TrainConfig, rng, initial_lora=None):
    if initial_lora is not None:
        lora_params = initial_lora
    else:
        lora_params = lora_mod.init_lora(
            params, cfg, rank=train_config.rank, alpha=train_config.alpha, rng=rng, dtype=jnp.float32
        )
    optimizer = make_optimizer(train_config)
    state = {
        "lora": lora_params,
        "opt_state": optimizer.init(lora_params),
        "step": jnp.zeros((), dtype=jnp.int32),
    }
    return state, optimizer


def make_accumulated_train_step(cfg: MossConfig, train_config: TrainConfig, optimizer, accumulation_steps: int):
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")

    def loss_fn(lora_params, params, latents, noise, sigmas, context, mask, weighting):
        noisy = noisy_model_input(latents, noise, sigmas)
        prediction = model_mod.forward(
            params,
            noisy,
            sigmas * NUM_TRAIN_TIMESTEPS,
            context,
            mask,
            cfg,
            lora=lora_params,
            remat=train_config.remat,
            query_chunk_size=train_config.query_chunk_size,
        )
        return rectified_flow_loss(
            prediction,
            latents,
            noise,
            sigmas,
            weighting,
            loss_type=train_config.loss_type,
            huber_c=train_config.huber_c,
        )

    def mean_loss_and_grads(lora_params, params, batches):
        def body(carry, batch):
            loss, grads = jax.value_and_grad(loss_fn)(lora_params, params, *batch)
            accum, total = carry
            accum = jax.tree.map(jnp.add, accum, grads)
            return (accum, total + loss), None

        zeros = jax.tree.map(jnp.zeros_like, lora_params)
        (accum, total), _ = jax.lax.scan(body, (zeros, jnp.zeros((), jnp.float32)), batches)
        return total / accumulation_steps, jax.tree.map(lambda x: x / accumulation_steps, accum)

    @jax.jit
    def train_step(state, params, batches):
        loss, grads = mean_loss_and_grads(state["lora"], params, batches)
        updates, opt_state = optimizer.update(grads, state["opt_state"], state["lora"])
        lora_params = optax.apply_updates(state["lora"], updates)
        new_state = {"lora": lora_params, "opt_state": opt_state, "step": state["step"] + 1}
        metrics = {"loss": loss, "grad_norm": optax.global_norm(grads)}
        return new_state, metrics

    return train_step


def make_train_step(cfg: MossConfig, train_config: TrainConfig, optimizer):
    def loss_fn(lora_params, params, latents, noise, sigmas, context, mask, weighting):
        noisy = noisy_model_input(latents, noise, sigmas)
        prediction = model_mod.forward(
            params,
            noisy,
            sigmas * NUM_TRAIN_TIMESTEPS,
            context,
            mask,
            cfg,
            lora=lora_params,
            remat=train_config.remat,
            query_chunk_size=train_config.query_chunk_size,
        )
        return rectified_flow_loss(
            prediction,
            latents,
            noise,
            sigmas,
            weighting,
            loss_type=train_config.loss_type,
            huber_c=train_config.huber_c,
        )

    @jax.jit
    def train_step(state, params, latents, noise, sigmas, context, mask, weighting):
        loss, grads = jax.value_and_grad(loss_fn)(state["lora"], params, latents, noise, sigmas, context, mask, weighting)
        updates, opt_state = optimizer.update(grads, state["opt_state"], state["lora"])
        lora_params = optax.apply_updates(state["lora"], updates)
        new_state = {"lora": lora_params, "opt_state": opt_state, "step": state["step"] + 1}
        metrics = {"loss": loss, "grad_norm": optax.global_norm(grads)}
        return new_state, metrics

    return train_step
