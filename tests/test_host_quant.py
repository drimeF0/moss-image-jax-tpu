import numpy as np
import jax.numpy as jnp
import torch
from safetensors.torch import save_file

from mossjax import quant, weights as jw
from mossjax import text_encoder as jte

from helpers import CFG, _params, _torch_model
from test_text_encoder import CFG as TE_CFG, _params as _te_params, _torch_model as _te_torch_model


def test_dit_host_quantization_matches_device(tmp_path):
    model = _torch_model()
    path = tmp_path / "dit.safetensors"
    save_file({key: value for key, value in model.state_dict().items()}, str(path))
    loaded = jw.load_params(str(path), CFG, quantize_int8=True)
    device = quant.quantize_params(_params(model), CFG)
    flat_loaded, flat_device = jw.flatten_tree(loaded), jw.flatten_tree(device)
    assert set(flat_loaded) == set(flat_device)
    for key in flat_loaded:
        a, b = flat_loaded[key], flat_device[key]
        if key.endswith(".int8"):
            diff = np.abs(np.asarray(a).astype(np.int32) - np.asarray(b).astype(np.int32))
            assert diff.max() <= 1
            assert float((diff > 0).mean()) < 1e-3
        elif key.endswith(".scale"):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=0)
        elif key.endswith(".act_dtype"):
            assert np.asarray(a).dtype == np.asarray(b).dtype
        else:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_te_host_quantization_matches_device(tmp_path):
    model = _te_torch_model()
    path = tmp_path / "te.safetensors"
    save_file({f"model.language_model.{k}": v for k, v in model.state_dict().items()}, str(path))
    loaded = jte.load_params(str(path), TE_CFG, quantize_int8=True)
    device = quant.quantize_text_encoder_params(_te_params(model))
    flat_loaded = jw.flatten_tree(loaded)
    flat_device = jw.flatten_tree(device)
    assert set(flat_loaded) == set(flat_device)
    quantized = 0
    for key in flat_loaded:
        a, b = flat_loaded[key], flat_device[key]
        if key.endswith(".int8"):
            quantized += 1
            diff = np.abs(np.asarray(a).astype(np.int32) - np.asarray(b).astype(np.int32))
            assert diff.max() <= 1
            assert float((diff > 0).mean()) < 1e-3
        elif key.endswith(".scale"):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=0)
        elif key.endswith(".act_dtype"):
            assert np.asarray(a).dtype == np.asarray(b).dtype
        else:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert quantized > 0
