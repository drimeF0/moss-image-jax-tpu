import json
import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
from safetensors.numpy import load_file
from safetensors.torch import save_file as torch_save_file

from mossjax import prepare, quant
from mossjax import text_encoder as jte
from mossjax import weights as jw

from helpers import CFG, _params, _torch_model
from test_text_encoder import CFG as TE_CFG, _params as _te_params, _torch_model as _te_torch_model


def _save_dit(tmp_path, name="dit.safetensors"):
    model = _torch_model()
    path = tmp_path / name
    torch_save_file({f"model.diffusion_model.{key}": value for key, value in model.state_dict().items()}, str(path))
    return path


def _assert_trees_equal(a, b):
    flat_a, flat_b = jw.flatten_tree(a), jw.flatten_tree(b)
    assert set(flat_a) == set(flat_b)
    for key in flat_a:
        np.testing.assert_array_equal(np.asarray(flat_a[key]), np.asarray(flat_b[key]))


def test_prepare_dit_matches_host_quantization(tmp_path):
    source = _save_dit(tmp_path)
    output = tmp_path / "prepared_dit.safetensors"
    stats = prepare.prepare_dit([str(source)], str(output))
    assert stats["quantized_keys"] == CFG.num_layers * 6
    assert stats["act_dtype"] == "float32"
    assert stats["bytes"] < stats["source_bytes"]

    prepared = jw.load_params(str(output), CFG)
    expected = jw.load_params(str(source), CFG, quantize_int8=True)
    _assert_trees_equal(prepared, expected)

    raw = load_file(str(output))
    assert raw["layers.0.proj.q.weight"].dtype == np.int8
    assert raw["layers.0.proj.q.weight.scale"].dtype == np.float32
    assert "layers.0.norm1.weight" in raw and "layers.0.norm1.weight.scale" not in raw
    assert raw["text_embedder.proj.weight"].dtype == np.float32
    metadata = jw.read_safetensors_metadata(str(output))
    assert metadata[quant.META_FORMAT] == quant.FORMAT_INT8_SCALE
    assert metadata[quant.META_ACT_DTYPE] == "float32"


def test_prepare_dit_with_fused_qkv(tmp_path):
    model = _torch_model()
    sd = {f"model.diffusion_model.{key}": value for key, value in model.state_dict().items()}
    q = sd.pop("model.diffusion_model.layers.1.proj.q.weight")
    k = sd.pop("model.diffusion_model.layers.1.proj.k.weight")
    v = sd.pop("model.diffusion_model.layers.1.proj.v.weight")
    sd["diffusion_model.layers.1.proj.qkv.weight"] = torch.cat([q, k, v], dim=0)
    source = tmp_path / "fused.safetensors"
    torch_save_file(sd, str(source))

    output = tmp_path / "prepared_fused.safetensors"
    prepare.prepare_dit([str(source)], str(output), cfg=CFG)
    prepared = jw.load_params(str(output), CFG)
    expected = jw.load_params(str(source), CFG, quantize_int8=True)
    _assert_trees_equal(prepared, expected)


def test_prepare_text_encoder_matches_host_quantization(tmp_path):
    model = _te_torch_model()
    source = tmp_path / "te.safetensors"
    torch_save_file(
        {f"model.language_model.{key}": value for key, value in model.state_dict().items()}, str(source)
    )
    output = tmp_path / "prepared_te.safetensors"
    stats = prepare.prepare_text_encoder([str(source)], TE_CFG, str(output))
    assert stats["quantized_keys"] > 0
    assert stats["act_dtype"] == "float32"

    prepared = jte.load_params(str(output), TE_CFG)
    expected = jte.load_params(str(source), TE_CFG, quantize_int8=True)
    _assert_trees_equal(prepared, expected)

    raw = load_file(str(output))
    assert raw["layers.0.linear_attn.in_proj_qkv.weight"].dtype == np.int8
    assert "layers.0.linear_attn.in_proj_b.weight" in raw
    assert "layers.0.linear_attn.in_proj_b.weight.scale" not in raw


def test_prepare_bf16_act_dtype(tmp_path):
    model = _torch_model().to(torch.bfloat16)
    source = tmp_path / "dit_bf16.safetensors"
    torch_save_file({key: value for key, value in model.state_dict().items()}, str(source))
    output = tmp_path / "prepared_bf16.safetensors"
    stats = prepare.prepare_dit([str(source)], str(output))
    assert stats["act_dtype"] == "bfloat16"

    params = jw.load_params(str(output), CFG)
    flat = jw.flatten_tree(params)
    assert flat["layers.0.proj.q.weight.act_dtype"].dtype == jnp.bfloat16
    assert flat["layers.0.proj.o.weight.int8"].dtype == jnp.int8
    assert flat["layers.0.norm1.weight"].dtype == jnp.bfloat16
    assert flat["layers.0.proj.q.weight.scale"].dtype == jnp.float32


def test_prepare_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = _save_dit(tmp_path)
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "prepare_quantized.py"),
            "--dit",
            str(source),
            "--output-dir",
            str(tmp_path / "prepared"),
            "--name",
            "tiny",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(root),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "prepared" / "tiny_dit_int8.safetensors").exists()
    summary = json.loads((tmp_path / "prepared" / "prepare_summary.json").read_text())
    assert summary["dit"]["quantized_keys"] == CFG.num_layers * 6


def test_cache_progress_reporting(capsys, tmp_path, monkeypatch):
    from PIL import Image

    from mossjax import caching

    monkeypatch.setattr(caching, "write_latents_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(caching, "write_te_cache", lambda *args, **kwargs: None)

    class Tokenizer:
        def __call__(self, text):
            return {"input_ids": list(range(10))}

    prompts = {"a.png": ["one"], "b.png": ["two captions", "second"]}
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rng = np.random.default_rng(0)
    for name in prompts:
        Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8")).save(images_dir / name)

    cached = caching.build_caches_from_images(
        str(images_dir),
        prompts,
        None,
        None,
        None,
        None,
        Tokenizer(),
        str(tmp_path / "cache"),
    )
    out = capsys.readouterr().out
    assert len(cached) == 2
    assert "[cache 1/2] a.png" in out
    assert "[cache 2/2] b.png" in out
    assert "eta" in out
    assert "[cache] done: 2 images" in out


def test_cache_progress_can_be_disabled(capsys, tmp_path, monkeypatch):
    from PIL import Image

    from mossjax import caching

    monkeypatch.setattr(caching, "write_latents_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(caching, "write_te_cache", lambda *args, **kwargs: None)

    class Tokenizer:
        def __call__(self, text):
            return {"input_ids": list(range(4))}

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.fromarray(np.zeros((32, 32, 3), dtype="uint8")).save(images_dir / "a.png")
    caching.build_caches_from_images(
        str(images_dir),
        {"a.png": ["one"]},
        None,
        None,
        None,
        None,
        Tokenizer(),
        str(tmp_path / "cache"),
        progress=False,
    )
    assert capsys.readouterr().out == ""


def _split_shards(sd, tmp_path, name):
    keys = sorted(sd)
    half = len(keys) // 2
    first = tmp_path / f"{name}-00001-of-00002.safetensors"
    second = tmp_path / f"{name}-00002-of-00002.safetensors"
    torch_save_file({key: sd[key] for key in keys[:half]}, str(first))
    torch_save_file({key: sd[key] for key in keys[half:]}, str(second))
    return first


def test_dit_load_resolves_first_shard(tmp_path):
    model = _torch_model()
    sd = {f"model.diffusion_model.{key}": value for key, value in model.state_dict().items()}
    first = _split_shards(sd, tmp_path, "diffusion_pytorch_model")
    loaded = jw.load_params(str(first), CFG)
    _assert_trees_equal(loaded, _params(model))

    globbed = jw.load_params(str(tmp_path / "diffusion_pytorch_model-*.safetensors"), CFG)
    _assert_trees_equal(globbed, _params(model))


def test_te_load_resolves_first_shard(tmp_path):
    model = _te_torch_model()
    sd = {f"model.language_model.{key}": value for key, value in model.state_dict().items()}
    first = _split_shards(sd, tmp_path, "model")
    loaded = jte.load_params(str(first), TE_CFG)
    _assert_trees_equal(loaded, _te_params(model))


def test_encode_prompt_embeds_bucketed_padding_matches_unpadded():
    from mossjax import caching

    model = _te_torch_model()
    params = _te_params(model)
    ids = (np.arange(10) % (model.config.vocab_size - 2) + 1).astype(np.int64)

    padded = caching.encode_prompt_embeds(params, TE_CFG, [ids], jit=True)[0]
    eager = caching.encode_prompt_embeds(params, TE_CFG, [ids], jit=False)[0]
    assert padded.shape == (10, model.config.hidden_size)
    np.testing.assert_allclose(padded, eager, rtol=1e-4, atol=1e-5)


def test_build_caches_uses_fixed_resolution_and_original_size(tmp_path, monkeypatch):
    from PIL import Image

    from mossjax import caching

    captured = {}

    def fake_latents(path, *args, **kwargs):
        captured["latents_path"] = path
        captured["original_size"] = kwargs.get("original_size")

    monkeypatch.setattr(caching, "write_latents_cache", fake_latents)
    monkeypatch.setattr(caching, "write_te_cache", lambda *args, **kwargs: None)

    class Tokenizer:
        def __call__(self, text):
            return {"input_ids": [1, 2, 3]}

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.fromarray(np.zeros((100, 50, 3), dtype="uint8")).save(images_dir / "a.png")
    caching.build_caches_from_images(
        str(images_dir),
        {"a.png": ["one"]},
        None,
        None,
        None,
        None,
        Tokenizer(),
        str(tmp_path / "cache"),
        resolution=(64, 96),
    )
    assert captured["latents_path"].endswith("a_0064x0096_moss.npz")
    assert captured["original_size"] == [50, 100]


def test_cache_validity_helpers(tmp_path):
    from mossjax.data import cache as cache_mod

    latents = np.zeros((8, 12, 8), dtype=np.float32)
    latents_path = tmp_path / "a_0064x0096_moss.npz"
    cache_mod.save_latents(latents_path, latents, original_size=[64, 96], crop_ltrb=[0, 0, 64, 96])
    assert cache_mod.is_latents_cache_valid(str(latents_path), (64, 96))
    assert not cache_mod.is_latents_cache_valid(str(latents_path), (64, 64))
    assert not cache_mod.is_latents_cache_valid(str(tmp_path / "missing.npz"), (64, 96))

    embeds = np.zeros((5, 8), dtype=np.float32)
    uncond = np.zeros((2, 8), dtype=np.float32)
    te_path = tmp_path / "a_moss_te.npz"
    cache_mod.save_te_outputs(te_path, [(embeds, "one"), (embeds, "two")], uncond)
    assert cache_mod.is_te_cache_valid(str(te_path), ["one", "two"])
    assert not cache_mod.is_te_cache_valid(str(te_path), ["one"])
    assert not cache_mod.is_te_cache_valid(str(te_path), ["one", "other"])
    assert cache_mod.is_te_cache_valid(str(te_path))


def test_build_caches_skips_existing_and_force_rebuilds(tmp_path, monkeypatch):
    from PIL import Image

    from mossjax import caching
    from mossjax.data import cache as cache_mod

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.fromarray(np.zeros((32, 32, 3), dtype="uint8")).save(images_dir / "a.png")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    cache_mod.save_latents(
        str(cache_dir / "a_0032x0032_moss.npz"),
        np.zeros((8, 4, 4), dtype=np.float32),
        original_size=[32, 32],
        crop_ltrb=[0, 0, 32, 32],
    )
    cache_mod.save_te_outputs(
        str(cache_dir / "a_moss_te.npz"),
        [(np.zeros((4, 8), dtype=np.float32), "one")],
        np.zeros((2, 8), dtype=np.float32),
    )

    calls = {"latents": 0, "te": 0}
    monkeypatch.setattr(caching, "write_latents_cache", lambda *args, **kwargs: calls.__setitem__("latents", calls["latents"] + 1))
    monkeypatch.setattr(caching, "write_te_cache", lambda *args, **kwargs: calls.__setitem__("te", calls["te"] + 1))

    class Tokenizer:
        def __call__(self, text):
            return {"input_ids": [1, 2, 3]}

    cached = caching.build_caches_from_images(
        str(images_dir),
        {"a.png": ["one"]},
        None,
        None,
        None,
        None,
        Tokenizer(),
        str(cache_dir),
        resolution=(32, 32),
        progress=False,
    )
    assert cached == ["a.png"]
    assert calls == {"latents": 0, "te": 0}

    caching.build_caches_from_images(
        str(images_dir),
        {"a.png": ["one"]},
        None,
        None,
        None,
        None,
        Tokenizer(),
        str(cache_dir),
        resolution=(32, 32),
        overwrite_existing=True,
        progress=False,
    )
    assert calls == {"latents": 1, "te": 1}


def test_build_caches_rebuilds_when_captions_changed(tmp_path, monkeypatch):
    from PIL import Image

    from mossjax import caching
    from mossjax.data import cache as cache_mod

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.fromarray(np.zeros((32, 32, 3), dtype="uint8")).save(images_dir / "a.png")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_mod.save_latents(
        str(cache_dir / "a_0032x0032_moss.npz"),
        np.zeros((8, 4, 4), dtype=np.float32),
        original_size=[32, 32],
        crop_ltrb=[0, 0, 32, 32],
    )
    cache_mod.save_te_outputs(
        str(cache_dir / "a_moss_te.npz"),
        [(np.zeros((4, 8), dtype=np.float32), "old caption")],
        np.zeros((2, 8), dtype=np.float32),
    )

    calls = {"te": 0}
    monkeypatch.setattr(caching, "write_latents_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(caching, "write_te_cache", lambda *args, **kwargs: calls.__setitem__("te", calls["te"] + 1))

    class Tokenizer:
        def __call__(self, text):
            return {"input_ids": [1, 2, 3]}

    caching.build_caches_from_images(
        str(images_dir),
        {"a.png": ["new caption"]},
        None,
        None,
        None,
        None,
        Tokenizer(),
        str(cache_dir),
        resolution=(32, 32),
        progress=False,
    )
    assert calls["te"] == 1
