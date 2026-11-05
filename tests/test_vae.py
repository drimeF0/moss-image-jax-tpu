import numpy as np
import jax.numpy as jnp
import torch
from safetensors.torch import load_file, save_file

from library import flux2_vae

from mossjax import vae as jv


def _torch_vae():
    torch.manual_seed(0)
    return flux2_vae.AutoEncoder(flux2_vae.AutoEncoderParams(resolution=32, ch=32, z_channels=32)).eval()


def _state_dict(model):
    return {key: value.detach().numpy() for key, value in model.state_dict().items()}


def _params(model):
    return jv.state_dict_to_params(_state_dict(model), jv.VaeConfig(ch=32, z_channels=32))


def test_encode_parity():
    model = _torch_vae()
    params = _params(model)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = model.encode(x).numpy()
    actual = np.asarray(jv.encode(params, jnp.asarray(x.numpy()), jv.VaeConfig(ch=32, z_channels=32)))
    assert actual.shape == expected.shape
    assert float(np.abs(actual - expected).max()) < 1e-5


def test_decode_parity():
    model = _torch_vae()
    params = _params(model)
    z = torch.randn(2, 32, 4, 4)
    with torch.no_grad():
        expected = model.decode(z).numpy()
    actual = np.asarray(jv.decode(params, jnp.asarray(z.numpy()), jv.VaeConfig(ch=32, z_channels=32)))
    assert actual.shape == expected.shape
    assert float(np.abs(actual - expected).max()) < 5e-5


def test_patchify_and_normalize_parity():
    torch.manual_seed(1)
    z = torch.randn(2, 32, 8, 6)
    np.testing.assert_allclose(np.asarray(jv.patchify(jnp.asarray(z.numpy()))), flux2_vae.patchify_latents(z).numpy(), atol=0)
    p = flux2_vae.patchify_latents(z)
    np.testing.assert_allclose(
        np.asarray(jv.unpatchify(jnp.asarray(p.numpy()))), flux2_vae.unpatchify_latents(p).numpy(), atol=0
    )
    mean = torch.randn(128)
    var = torch.rand(128) + 0.5
    expected = flux2_vae.normalize_latents(p, mean, var)
    actual = jv.normalize(jnp.asarray(p.numpy()), jnp.asarray(mean.numpy()), jnp.asarray(var.numpy()))
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), atol=1e-6)
    back = jv.denormalize(actual, jnp.asarray(mean.numpy()), jnp.asarray(var.numpy()))
    np.testing.assert_allclose(np.asarray(back), p.numpy(), atol=1e-6)


def test_encode_for_dit_parity():
    model = _torch_vae()
    params = _params(model)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = model.encode_for_dit(x).numpy()
    actual = np.asarray(jv.encode_for_dit(params, jnp.asarray(x.numpy()), jv.VaeConfig(ch=32, z_channels=32)))
    assert actual.shape == expected.shape
    assert float(np.abs(actual - expected).max()) < 1e-5


def test_decode_from_dit_parity():
    model = _torch_vae()
    params = _params(model)
    z = torch.randn(1, 128, 2, 2)
    with torch.no_grad():
        expected = model.decode_from_dit(z).numpy()
    actual = np.asarray(jv.decode_from_dit(params, jnp.asarray(z.numpy()), jv.VaeConfig(ch=32, z_channels=32)))
    assert actual.shape == expected.shape
    assert float(np.abs(actual - expected).max()) < 5e-5


def test_load_from_safetensors_roundtrip(tmp_path):
    model = _torch_vae()
    path = tmp_path / "vae.safetensors"
    save_file({key: torch.from_numpy(np.array(value)) for key, value in _state_dict(model).items()}, str(path))
    sd = {key: value.numpy() for key, value in load_file(str(path)).items()}
    params, cfg = jv.load_vae_params(sd, jv.VaeConfig(ch=32, z_channels=32))
    assert cfg.decoder_ch is None
    reference = _params(model)
    from mossjax import weights as w

    flat_a, flat_b = w.flatten_tree(params), w.flatten_tree(reference)
    assert set(flat_a) == set(flat_b)
    for key in flat_a:
        np.testing.assert_array_equal(np.asarray(flat_a[key]), np.asarray(flat_b[key]))


def test_diffusers_conversion_matches_upstream():
    torch.manual_seed(2)
    shapes = {
        "quant_conv.weight": (8, 8, 1, 1),
        "post_quant_conv.bias": (8,),
        "encoder.down_blocks.0.resnets.0.conv_shortcut.weight": (4, 2, 1, 1),
        "encoder.down_blocks.1.downsamplers.0.conv.weight": (4, 4, 3, 3),
        "encoder.conv_norm_out.weight": (4,),
        "decoder.up_blocks.1.resnets.2.conv_shortcut.bias": (4,),
        "decoder.up_blocks.0.upsamplers.0.conv.weight": (4, 4, 3, 3),
        "decoder.conv_norm_out.bias": (4,),
        "decoder.mid_block.attentions.0.group_norm.weight": (4,),
        "decoder.mid_block.attentions.0.to_q.weight": (4, 4),
        "decoder.mid_block.attentions.0.to_out.0.bias": (4,),
        "decoder.mid_block.resnets.1.conv1.weight": (4, 4, 3, 3),
        "decoder.up_blocks.1.resnets.0.norm1.weight": (4,),
    }
    tensors = {key: torch.randn(*shape) for key, shape in shapes.items()}
    expected = flux2_vae._convert_diffusers_vae_state_dict(tensors)
    actual = jv.convert_diffusers_state_dict({key: value.numpy() for key, value in tensors.items()})
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_allclose(np.asarray(actual[key]), expected[key].numpy(), atol=0)
