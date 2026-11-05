import jax.numpy as jnp
import torch

from library import moss_image_models

from mossjax import weights as jw
from mossjax.config import MossConfig

SMALL_CONFIG = dict(
    dim=64,
    ffn_dim=96,
    num_heads=4,
    num_layers=3,
    text_dim=24,
    freq_dim=16,
    latent_channels=8,
    rope_base=10000.0,
)
CFG = MossConfig(**SMALL_CONFIG)


def _torch_model():
    torch.manual_seed(0)
    return moss_image_models.MossImage(**SMALL_CONFIG, attn_mode="torch", split_attn=False).eval()


def _params(model):
    return jw.state_dict_to_params({key: value.numpy() for key, value in model.state_dict().items()}, CFG)


def _inputs():
    torch.manual_seed(1)
    batch, height, width = 2, 3, 5
    lens = [4, 7]
    x = torch.randn(batch, SMALL_CONFIG["latent_channels"], height, width)
    context = torch.randn(batch, max(lens), SMALL_CONFIG["text_dim"])
    mask = torch.zeros(batch, max(lens), dtype=torch.int32)
    for i, length in enumerate(lens):
        mask[i, :length] = 1
    context = context * mask[..., None]
    timesteps = torch.tensor([750.0, 120.0])
    return x, timesteps, context, mask


def _jnp(tree):
    return jnp.asarray(tree.numpy())
