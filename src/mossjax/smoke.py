import jax

from . import init, text_encoder as text_encoder_mod, vae as vae_mod, weights
from .config import MossConfig
from .text_encoder import TextConfig
from .vae import VaeConfig

SMOKE_DIT = MossConfig(
    dim=64,
    ffn_dim=96,
    num_heads=4,
    num_layers=2,
    text_dim=32,
    freq_dim=16,
    latent_channels=32,
    rope_base=10000.0,
)

SMOKE_TE = TextConfig(
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    vocab_size=64,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    partial_rotary_factor=0.25,
    mrope_section=(1, 0, 0),
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_conv_kernel_dim=4,
    layer_types=("linear_attention", "full_attention"),
    hidden_act="silu",
    attention_bias=False,
)

SMOKE_VAE = VaeConfig(ch=32, z_channels=8)
SMOKE_IMAGE_SIZE = 32


def build_params(seed: int = 0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    dit = init.init_tree(weights.param_shapes(SMOKE_DIT), keys[0], ("layers", "blocks"))
    text_encoder = init.init_tree(text_encoder_mod.param_shapes(SMOKE_TE), keys[1], ("layers",))
    vae = init.init_tree(vae_mod.param_shapes(SMOKE_VAE), keys[2], vae_mod.VAE_LIST_CONTAINERS)
    return dit, text_encoder, vae
