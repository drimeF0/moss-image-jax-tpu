import numpy as np
import jax.numpy as jnp
import torch

from library import flux2_vae, moss_image_models

from networks import lora_generic

from mossjax import caching, lora, trainer, train as train_mod
from mossjax import text_encoder as jte
from mossjax import vae as jv
from mossjax import weights as jw
from mossjax.config import MossConfig
from mossjax.data import cache as cache_mod

DIT_CONFIG = dict(
    dim=64,
    ffn_dim=96,
    num_heads=4,
    num_layers=2,
    text_dim=32,
    freq_dim=16,
    latent_channels=32,
    rope_base=10000.0,
)
DIT = MossConfig(**DIT_CONFIG)
TE_TINY = dict(
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
    mrope_section=[1, 0, 0],
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_conv_kernel_dim=4,
    layer_types=["linear_attention", "full_attention"],
    hidden_act="silu",
    attention_bias=False,
)
TE_CFG = jte.config_from_hf(TE_TINY)
VAE_CFG = jv.VaeConfig(ch=32, z_channels=8)
BUCKET = (32, 32)


def _build_params():
    torch.manual_seed(0)
    dit = moss_image_models.MossImage(**DIT_CONFIG, attn_mode="torch").eval()
    dit_params = jw.state_dict_to_params({key: value.numpy() for key, value in dit.state_dict().items()}, DIT)

    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    te = Qwen3_5TextModel(Qwen3_5TextConfig(**TE_TINY)).eval()
    te_params = jte.state_dict_to_params({key: value.detach().numpy() for key, value in te.state_dict().items()}, TE_CFG)

    torch.manual_seed(1)
    vae = flux2_vae.AutoEncoder(flux2_vae.AutoEncoderParams(resolution=32, ch=32, z_channels=8)).eval()
    vae_params = jv.state_dict_to_params(
        {key: value.detach().numpy() for key, value in vae.state_dict().items()}, VAE_CFG
    )
    return dit_params, te_params, vae_params


def test_e2e_cache_train_export(tmp_path):
    caches = tmp_path / "cache"
    caches.mkdir()
    dit_params, te_params, vae_params = _build_params()
    rng = np.random.default_rng(2)
    for index in range(2):
        image = rng.uniform(-1.0, 1.0, (3, 32, 32)).astype(np.float32)
        latents_path = caches / f"image{index}_0032x0032_moss.npz"
        caching.write_latents_cache(str(latents_path), vae_params, VAE_CFG, image, original_size=[32, 32])
        te_path = caches / f"image{index}_moss_te.npz"
        caption_ids = rng.integers(0, TE_TINY["vocab_size"], size=9)
        uncond_ids = rng.integers(0, TE_TINY["vocab_size"], size=3)
        caching.write_te_cache(
            str(te_path), te_params, TE_CFG, [caption_ids], [f"caption {index}"], uncond_ids, caption_dropout_rate=0.5
        )
    records = cache_mod.find_cache_pairs(str(caches))
    assert len(records) == 2

    vae_stats = {
        "running_mean": np.asarray(vae_params["bn"]["running_mean"]),
        "running_var": np.asarray(vae_params["bn"]["running_var"]),
    }
    train_config = trainer.TrainConfig(rank=2, learning_rate=5e-2, remat=True)
    loop_config = train_mod.LoopConfig(steps=8, timestep_sampling="uniform", log_every=0, save_every=4, seed=0)
    output_dir = tmp_path / "out"
    messages = []
    state, history = train_mod.train_loop(
        dit_params,
        DIT,
        records,
        train_config,
        loop_config,
        bucket_reso=BUCKET,
        output_dir=str(output_dir),
        vae_stats=vae_stats,
        log_fn=messages.append,
    )
    assert any(message.startswith("[train] new input shape #1") for message in messages)
    assert len(history) == 8
    assert all(np.isfinite(history))
    final_path = output_dir / "moss_lora_final.safetensors"
    assert final_path.exists()
    assert (output_dir / "lora_step0000004.safetensors").exists()

    back = lora.load_lora_file(str(final_path), dit_params, DIT)
    assert jnp.abs(lora.merge_into_params(dit_params, back, DIT)["layers"][0]["proj"]["q"]["weight"]).sum() > 0

    torch.manual_seed(3)
    model = moss_image_models.MossImage(**DIT_CONFIG, attn_mode="torch").eval()
    network, weights_sd = lora_generic.create_network_from_weights(1.0, str(final_path), None, [], model)
    network.apply_to([], model)
    info = network.load_state_dict(weights_sd, strict=True)
    assert not info.missing_keys and not info.unexpected_keys
    with torch.no_grad():
        x = torch.randn(1, DIT_CONFIG["latent_channels"], 2, 2)
        context = torch.randn(1, 16, DIT_CONFIG["text_dim"])
        mask = torch.ones(1, 16, dtype=torch.int32)
        out = model(x, torch.tensor([500.0]), context, mask)
    assert out.shape == x.shape and torch.isfinite(out).all()
