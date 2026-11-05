import numpy as np
import jax
import jax.numpy as jnp
import pytest
import torch
from safetensors.torch import save_file

from mossjax import lora
from mossjax import model as jm

from helpers import CFG, _inputs, _jnp, _params, _torch_model

RANK = 2
ALPHA = 2.0


def _tree(params):
    return lora.init_lora(params, CFG, rank=RANK, alpha=ALPHA, rng=jax.random.PRNGKey(0))


def test_export_keys_match_kohya_layout():
    params = _params(_torch_model())
    tree = _tree(params)
    sd = lora.export_lora(tree, CFG, dim=RANK, alpha=ALPHA)
    expected = set()
    for path in lora.target_paths(CFG):
        name = lora.module_name(path)
        expected |= {f"{name}.lora_down.weight", f"{name}.lora_up.weight", f"{name}.alpha"}
    assert set(sd) == expected
    assert len([key for key in sd if key.endswith("lora_down.weight")]) == CFG.num_layers * len(lora.TARGET_SUFFIXES)
    assert "lora_unet_layers_0_proj_q.lora_down.weight" in sd
    assert "lora_unet_layers_0_proj_ffn_up_gate.lora_up.weight" in sd


def test_import_roundtrip():
    params = _params(_torch_model())
    tree = _tree(params)
    sd = lora.export_lora(tree, CFG, dim=RANK, alpha=ALPHA)
    back = lora.import_lora(sd, params, CFG)
    for index, block in enumerate(tree["layers"]):
        for suffix in lora.TARGET_SUFFIXES:
            name = lora._suffix_name(suffix)
            first, second = block["proj"][name], back["layers"][index]["proj"][name]
            np.testing.assert_array_equal(np.asarray(first["A"]), np.asarray(second["A"]))
            np.testing.assert_array_equal(np.asarray(first["B"]), np.asarray(second["B"]))
            np.testing.assert_allclose(float(first["scale"]), float(second["scale"]))


def test_import_rejects_unknown_module():
    with pytest.raises(ValueError, match="unknown LoRA module"):
        lora.import_lora({"lora_unet_layers_0_proj_unknown.lora_down.weight": np.zeros((2, 2))}, None, CFG)


def test_lora_forward_equals_merged_weights():
    params = _params(_torch_model())
    tree = _tree(params)
    x, timesteps, context, mask = _inputs()
    with_lora = np.asarray(jm.forward(params, _jnp(x), _jnp(timesteps), _jnp(context), _jnp(mask), CFG, lora=tree))
    merged = lora.merge_into_params(params, tree, CFG)
    merged_out = np.asarray(jm.forward(merged, _jnp(x), _jnp(timesteps), _jnp(context), _jnp(mask), CFG))
    np.testing.assert_allclose(with_lora, merged_out, atol=1e-5, rtol=0)


def test_upstream_loads_exported_lora_and_forward_matches(tmp_path):
    from networks import lora_generic

    model = _torch_model()
    params = _params(model)
    tree = _tree(params)
    x, timesteps, context, mask = _inputs()
    actual = np.asarray(jm.forward(params, _jnp(x), _jnp(timesteps), _jnp(context), _jnp(mask), CFG, lora=tree))

    sd = lora.export_lora(tree, CFG, dim=RANK, alpha=ALPHA)
    path = tmp_path / "moss_lora.safetensors"
    save_file({key: torch.from_numpy(value) for key, value in sd.items()}, str(path), lora.export_metadata(RANK, ALPHA))

    network, weights_sd = lora_generic.create_network_from_weights(1.0, str(path), None, [], model)
    network.apply_to([], model)
    info = network.load_state_dict(weights_sd, strict=True)
    assert not info.missing_keys and not info.unexpected_keys
    assert len(network.unet_loras) == CFG.num_layers * len(lora.TARGET_SUFFIXES)

    first = network.unet_loras[0]
    leaf = tree["layers"][0]["proj"]["q"]
    assert first.lora_name == "lora_unet_layers_0_proj_q"
    assert torch.allclose(first.lora_down.weight, torch.from_numpy(np.asarray(leaf["A"])))
    assert torch.allclose(first.lora_up.weight, torch.from_numpy(np.asarray(leaf["B"])))
    assert float(first.alpha) == ALPHA

    with torch.no_grad():
        expected = model(x, timesteps, context, mask).numpy()
    assert np.abs(actual - expected).max() < 1e-5
