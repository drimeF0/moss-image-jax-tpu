import jax
import jax.numpy as jnp
import numpy as np

from mossjax import model as jm
from mossjax import quant

from helpers import CFG, _inputs, _jnp, _params, _torch_model


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_quantize_weight_error_bound():
    weight = jax.random.normal(jax.random.PRNGKey(0), (16, 37), dtype=jnp.float32) * 2.0
    quantized, scale = quant.quantize_weight(weight)
    dequantized = quant.dequantize_weight(quantized, scale)
    assert quantized.dtype == jnp.int8
    assert quantized.shape == weight.shape
    assert scale.shape == (16,)
    error = jnp.abs(dequantized - weight)
    assert bool(jnp.all(error <= scale[:, None] * 0.5 * (1 + 1e-6)))


def test_quantize_weight_handles_zero_row():
    weight = jnp.zeros((4, 8), dtype=jnp.float32)
    quantized, scale = quant.quantize_weight(weight)
    assert bool(jnp.all(quantized == 0))
    assert bool(jnp.all(jnp.isfinite(scale)))


def test_quantize_params_scope():
    params = _params(_torch_model())
    quantized = quant.quantize_params(params, CFG)
    for index, layer in enumerate(quantized["layers"]):
        for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
            assert isinstance(layer["proj"][name]["weight"], dict)
        assert jnp.array_equal(layer["norm1"]["weight"], params["layers"][index]["norm1"]["weight"])
        assert jnp.array_equal(layer["norm2"]["weight"], params["layers"][index]["norm2"]["weight"])
    assert jnp.array_equal(quantized["video_embedder"]["weight"], params["video_embedder"]["weight"])
    assert jnp.array_equal(quantized["text_embedder"]["proj"]["weight"], params["text_embedder"]["proj"]["weight"])
    assert jnp.array_equal(quantized["final_linear_video"]["weight"], params["final_linear_video"]["weight"])


def test_quantized_forward_close_to_float():
    params = _params(_torch_model())
    quantized = quant.quantize_params(params, CFG)
    x, timesteps, context, mask = _inputs()
    expected = np.asarray(jm.forward(params, _jnp(x), _jnp(timesteps), _jnp(context), _jnp(mask), CFG))
    actual = np.asarray(jm.forward(quantized, _jnp(x), _jnp(timesteps), _jnp(context), _jnp(mask), CFG))
    assert actual.shape == expected.shape
    assert cosine(actual, expected) > 0.9999
    max_diff = float(np.abs(actual - expected).max())
    assert max_diff < 0.05, max_diff


def test_int8_compute_matches_weight_only():
    weight = jax.random.normal(jax.random.PRNGKey(1), (24, 40), dtype=jnp.float32) * 0.5
    layer = quant.quantize_linear_layer(weight)
    x = jax.random.normal(jax.random.PRNGKey(2), (3, 5, 40), dtype=jnp.float32)
    a = quant.weight_only_linear(x, layer)
    b = quant.int8_compute_linear(x, layer)
    assert cosine(a, b) > 0.9999


def test_int8_storage_is_smaller():
    params = _params(_torch_model())
    quantized = quant.quantize_params(params, CFG)
    assert quant.tree_nbytes(quantized) < quant.tree_nbytes(params)
    original_layers = sum(quant.tree_nbytes(layer) for layer in params["layers"])
    quantized_layers = sum(quant.tree_nbytes(layer) for layer in quantized["layers"])
    assert quantized_layers / original_layers < 0.4


def test_int8_compute_preserves_act_dtype():
    weight = jax.random.normal(jax.random.PRNGKey(3), (16, 32), dtype=jnp.bfloat16)
    layer = quant.quantize_linear_layer(weight)
    x = jax.random.normal(jax.random.PRNGKey(4), (2, 32), dtype=jnp.bfloat16)
    y = quant.quantized_linear(x, layer, compute="int8")
    assert y.dtype == jnp.bfloat16
