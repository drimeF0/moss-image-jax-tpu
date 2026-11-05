import numpy as np
import jax
import jax.numpy as jnp
import pytest
import torch

from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from mossjax import text_encoder as jte

TINY = dict(
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    vocab_size=128,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    partial_rotary_factor=0.25,
    mrope_section=[2, 1, 1],
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    hidden_act="silu",
    attention_bias=False,
)
CFG = jte.config_from_hf(TINY)


def _torch_model():
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(**TINY)
    model = Qwen3_5TextModel(config)
    model.config._attn_implementation = "eager"
    return model.eval()


def _params(model):
    return jte.state_dict_to_params({key: value.detach().numpy() for key, value in model.state_dict().items()}, CFG)


def _inputs():
    torch.manual_seed(1)
    ids = torch.randint(0, TINY["vocab_size"], (2, 11))
    return ids


def test_forward_matches_transformers():
    model = _torch_model()
    params = _params(model)
    ids = _inputs()
    with torch.no_grad():
        expected = model(input_ids=ids).last_hidden_state.numpy()
    actual = np.asarray(jte.forward(params, jnp.asarray(ids.numpy()), CFG))
    assert actual.shape == expected.shape
    max_diff = float(np.abs(actual - expected).max())
    assert max_diff < 1e-4, f"max diff {max_diff}"


@pytest.mark.parametrize("layer_index", [0, 2])
def test_single_layer_matches_transformers(layer_index):
    model = _torch_model()
    params = _params(model)
    ids = _inputs()
    with torch.no_grad():
        x = model.embed_tokens(ids)
        position_ids = torch.arange(ids.shape[1]).view(1, 1, -1).expand(4, ids.shape[0], -1)
        embeddings = model.rotary_emb(x, position_ids[1:])
        output = model.layers[layer_index](
            x,
            position_embeddings=embeddings,
            attention_mask=None,
            past_key_values=None,
        )
        expected = (output[0] if isinstance(output, tuple) else output).numpy()
    batch, length = ids.shape
    position_ids_j = jnp.broadcast_to(jnp.arange(length), (batch, length))
    cos, sin = jte.rope_cos_sin(position_ids_j, CFG)
    x_j = jnp.asarray(x.numpy())
    actual = np.asarray(jte.decoder_layer(x_j, params["layers"][layer_index], cos, sin, CFG, layer_index))
    max_diff = float(np.abs(actual - expected).max())
    assert max_diff < 1e-4, f"max diff {max_diff}"


def test_l2norm_matches():
    x = jnp.asarray(np.random.RandomState(0).randn(3, 5).astype(np.float32))
    torch_x = torch.from_numpy(np.array(x)).float()
    inv = torch.rsqrt((torch_x * torch_x).sum(-1, keepdim=True) + 1e-6)
    expected = (torch_x * inv).numpy()
    np.testing.assert_allclose(np.asarray(jte.l2norm(x)), expected, atol=1e-6)


def test_real_config_parses():
    from pathlib import Path

    path = Path(__file__).resolve().parent / "reference" / "kohya_sd_scripts" / "configs" / "moss_image_qwen35_9b" / "config.json"
    cfg = jte.load_config(str(path))
    assert cfg.num_hidden_layers == 32
    assert cfg.head_dim == 256
    assert cfg.rotary_dim == 64
    assert cfg.layer_types.count("full_attention") == 8
    assert cfg.layer_types[3] == "full_attention"
    assert cfg.conv_dim == 2 * 16 * 128 + 32 * 128


def test_quantized_forward_close_to_float():
    from mossjax import quant

    model = _torch_model()
    params = _params(model)
    quantized = quant.quantize_text_encoder_params(params)
    assert isinstance(quantized["embed_tokens"]["weight"], dict)
    assert isinstance(quantized["layers"][0]["linear_attn"]["in_proj_qkv"]["weight"], dict)
    assert not isinstance(quantized["layers"][0]["linear_attn"]["in_proj_b"]["weight"], dict)
    assert isinstance(quantized["layers"][3]["self_attn"]["q_proj"]["weight"], dict)
    ids = _inputs()
    expected = np.asarray(jte.forward(params, jnp.asarray(ids.numpy()), CFG))
    actual = np.asarray(jte.forward(quantized, jnp.asarray(ids.numpy()), CFG))
    a = expected.astype(np.float64).ravel()
    b = actual.astype(np.float64).ravel()
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cosine > 0.9999, cosine
    assert float(np.abs(actual - expected).max()) < 0.1


def test_load_params_from_safetensors_skips_vision(tmp_path):
    from safetensors.torch import save_file

    model = _torch_model()
    sd = {f"model.language_model.{key}": value for key, value in model.state_dict().items()}
    sd["visual.patch_embed.weight"] = torch.zeros(3, 3)
    sd["lm_head.weight"] = torch.zeros(4, 4)
    path = tmp_path / "te.safetensors"
    save_file(sd, str(path))
    loaded = jte.load_params(str(path), CFG)
    reference = _params(model)
    for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(reference)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
