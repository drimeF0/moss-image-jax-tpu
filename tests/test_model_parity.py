import numpy as np
import jax.numpy as jnp
import pytest
import torch

from library import moss_image_models

from mossjax import model as jm
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


def test_forward_matches_vendored_torch():
    model = _torch_model()
    params = _params(model)
    x, timesteps, context, mask = _inputs()
    with torch.no_grad():
        expected = model(x, timesteps, context, mask).numpy()
    actual = np.asarray(jm.forward(params, _jnp(x), _jnp(timesteps), _jnp(context), _jnp(mask), CFG))
    max_diff = float(np.abs(actual - expected).max())
    assert actual.shape == expected.shape
    assert max_diff < 1e-6, f"max diff {max_diff}"


def test_forward_skip_layers_matches_torch():
    model = _torch_model()
    params = _params(model)
    x, timesteps, context, mask = _inputs()
    with torch.no_grad():
        expected = model(x, timesteps, context, mask, skip_layers=(1,)).numpy()
    actual = np.asarray(jm.forward(params, _jnp(x), _jnp(timesteps), _jnp(context), _jnp(mask), CFG, skip_layers=(1,)))
    assert float(np.abs(actual - expected).max()) < 1e-6


def test_forward_without_mask_is_full_length():
    model = _torch_model()
    params = _params(model)
    x, timesteps, context, mask = _inputs()
    with torch.no_grad():
        expected = model(x, timesteps, context, torch.ones_like(mask)).numpy()
    actual = np.asarray(jm.forward(params, _jnp(x), _jnp(timesteps), _jnp(context), None, CFG))
    assert float(np.abs(actual - expected).max()) < 1e-6


def test_padding_positions_do_not_matter():
    model = _torch_model()
    params = _params(model)
    x, timesteps, context, mask = _inputs()
    other = context.clone()
    other[mask == 0] = 7.0
    first = np.asarray(jm.forward(params, _jnp(x), _jnp(timesteps), _jnp(context), _jnp(mask), CFG))
    second = np.asarray(jm.forward(params, _jnp(x), _jnp(timesteps), _jnp(other), _jnp(mask), CFG))
    np.testing.assert_allclose(first, second, atol=1e-6, rtol=0)


def test_state_dict_roundtrip():
    model = _torch_model()
    sd = {key: value.numpy() for key, value in model.state_dict().items()}
    params = jw.state_dict_to_params(sd, CFG)
    back = jw.params_to_state_dict(params)
    for key, value in sd.items():
        if key in jw.DERIVED_KEYS:
            continue
        np.testing.assert_array_equal(back[key], value)


def test_split_fused_qkv_matches_upstream_order():
    torch.manual_seed(0)
    dim = SMALL_CONFIG["dim"]
    fused = torch.randn(3 * dim, dim)
    keys = jw.split_fused_qkv("layers.3.proj.qkv.weight", dim)
    assert keys == ["layers.3.proj.q.weight", "layers.3.proj.k.weight", "layers.3.proj.v.weight"]
    assert jw.split_fused_qkv("text_embedder.blocks.0.qkv.weight", dim) is None
    assert jw.split_fused_qkv("layers.3.proj.q.weight", dim) is None


def test_fused_qkv_state_dict_is_split_like_upstream():
    model = _torch_model()
    sd = {key: value.numpy() for key, value in model.state_dict().items()}
    q = sd.pop("layers.1.proj.q.weight")
    k = sd.pop("layers.1.proj.k.weight")
    v = sd.pop("layers.1.proj.v.weight")
    sd["layers.1.proj.qkv.weight"] = np.concatenate([q, k, v], axis=0)
    params = jw.state_dict_to_params(sd, CFG)
    flat = jw.flatten_tree(params)
    np.testing.assert_array_equal(np.asarray(flat["layers.1.proj.q.weight"]), q)
    np.testing.assert_array_equal(np.asarray(flat["layers.1.proj.k.weight"]), k)
    np.testing.assert_array_equal(np.asarray(flat["layers.1.proj.v.weight"]), v)


def test_state_dict_prefixes_are_stripped():
    model = _torch_model()
    sd = {f"model.diffusion_model.{key}": value.numpy() for key, value in model.state_dict().items()}
    params = jw.state_dict_to_params(sd, CFG)
    assert "layers" in params and "video_embedder" in params and "text_embedder" in params


def test_missing_key_raises():
    model = _torch_model()
    sd = {key: value.numpy() for key, value in model.state_dict().items()}
    sd.pop("layers.0.proj.q.weight")
    with pytest.raises(ValueError, match="missing keys"):
        jw.state_dict_to_params(sd, CFG)
