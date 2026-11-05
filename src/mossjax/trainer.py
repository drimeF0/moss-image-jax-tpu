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


def make_optimizer(train_config: TrainConfig):
    return optax.chain(
        optax.clip_by_global_norm(train_config.max_grad_norm),
        optax.adamw(
            learning_rate=train_config.learning_rate,
            b1=train_config.beta1,
            b2=train_config.beta2,
            eps=train_config.eps,
            weight_decay=train_config.weight_decay,
        ),
    )


def init_train_state(params, cfg: MossConfig, train_config: TrainConfig, rng):
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


def make_train_step(cfg: MossConfig, train_config: TrainConfig, optimizer):
    def loss_fn(lora_params, params, latents, noise, sigmas, context, mask, weighting):
        noisy = noisy_model_input(latents, noise, sigmas)
        prediction = model_mod.forward(
            params, noisy, sigmas * NUM_TRAIN_TIMESTEPS, context, mask, cfg, lora=lora_params, remat=train_config.remat
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
