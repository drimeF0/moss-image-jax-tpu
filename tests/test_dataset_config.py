import numpy as np

from mossjax.data import build_records, cache as cache_mod, load_dataset_config
from mossjax.data.dataset import CachedDataset

from helpers import CFG


def _write_record(directory, stem, caption_embeds, uncond_embeds, original_size, dropout=0.0, latents_shape=(32, 12, 8)):
    te_path = directory / f"{stem}_moss_te.npz"
    cache_mod.save_te_outputs(te_path, [(caption_embeds, "caption")], uncond_embeds, dropout)
    latents_path = directory / f"{stem}_{original_size[0]:04d}x{original_size[1]:04d}_moss.npz"
    latents = np.random.default_rng(0).standard_normal(latents_shape, dtype=np.float32)
    cache_mod.save_latents(latents_path, latents, original_size=list(original_size), crop_ltrb=[0, 0, *original_size])
    return te_path, latents_path


def test_load_dataset_config_parses_and_resolves_paths(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    config_path = tmp_path / "dataset.toml"
    config_path.write_text(
        """
[general]
caption_dropout_rate = 0.1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
resolution = 512
batch_size = 2
min_bucket_reso = 128
max_bucket_reso = 768
bucket_reso_steps = 32

  [[datasets.subsets]]
  image_dir = "images"
  num_repeats = 3

  [[datasets.subsets]]
  image_dir = "images"
  caption_dropout_rate = 0.5

[[datasets]]
resolution = "768,512"
enable_bucket = false
  [[datasets.subsets]]
  image_dir = "images"
""".lstrip()
    )
    config = load_dataset_config(str(config_path))
    assert len(config.datasets) == 2
    first, second = config.datasets
    assert first.resolution == (512, 512)
    assert first.batch_size == 2
    assert first.enable_bucket is True
    assert first.min_bucket_reso == 128 and first.max_bucket_reso == 768 and first.bucket_reso_steps == 32
    assert len(first.subsets) == 2
    assert first.subsets[0].image_dir == str(images)
    assert first.subsets[0].num_repeats == 3
    assert first.subsets[0].caption_dropout_rate == 0.1
    assert first.subsets[1].caption_dropout_rate == 0.5
    assert second.resolution == (768, 512)
    assert second.enable_bucket is False
    assert second.subsets[0].num_repeats == 1


def test_build_records_sets_repeats_and_bucket(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    embeds = np.random.default_rng(1).standard_normal((5, 8), dtype=np.float32)
    uncond = np.random.default_rng(2).standard_normal((3, 8), dtype=np.float32)
    _write_record(images, "img", embeds, uncond, original_size=(800, 600))
    config_path = tmp_path / "dataset.toml"
    config_path.write_text(
        """
[[datasets]]
resolution = 1024
enable_bucket = true
  [[datasets.subsets]]
  image_dir = "images"
  num_repeats = 2
""".lstrip()
    )
    records = build_records(load_dataset_config(str(config_path)))
    assert len(records) == 1
    assert records[0]["num_repeats"] == 2
    assert records[0]["bucket"] == (1024, 768)
    assert records[0]["subset"].num_repeats == 2


def test_cached_dataset_uses_repeats_weights_and_overrides(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(3)
    embed_a = rng.standard_normal((6, 8), dtype=np.float32)
    embed_b = rng.standard_normal((4, 8), dtype=np.float32)
    uncond = rng.standard_normal((2, 8), dtype=np.float32)
    te_a, _ = _write_record(images, "a", embed_a, uncond, original_size=(64, 96), dropout=0.0, latents_shape=(8, 12, 8))
    te_b, _ = _write_record(images, "b", embed_b, uncond, original_size=(64, 96), dropout=0.0, latents_shape=(8, 12, 8))

    records = [
        {
            "key": "a",
            "text_encoder": str(te_a),
            "latents": sorted(str(p) for p in images.glob("a_*_moss.npz")),
            "num_repeats": 3,
            "caption_dropout_rate": 1.0,
            "bucket": (64, 96),
        },
        {
            "key": "b",
            "text_encoder": str(te_b),
            "latents": sorted(str(p) for p in images.glob("b_*_moss.npz")),
            "num_repeats": 1,
            "caption_dropout_rate": None,
            "bucket": (64, 96),
        },
    ]
    vae_stats = {"running_mean": np.zeros(32, np.float32), "running_var": np.ones(32, np.float32)}
    dataset = CachedDataset(records, bucket_reso=(64, 96), seed=0, vae_stats=vae_stats)
    np.testing.assert_allclose(dataset.weights, [0.75, 0.25], rtol=0, atol=1e-12)

    sample = dataset.sample(index=0)
    assert sample["latents"].shape == (1, 32, 6, 4)
    np.testing.assert_array_equal(sample["context_mask"][0, :2], [1, 1])
    np.testing.assert_array_equal(sample["context_mask"][0, 2:], np.zeros(sample["context_mask"].shape[1] - 2, np.int32))

    uncond_bits = cache_mod.uint16_to_float32(cache_mod.float32_to_bf16_bits(uncond))
    np.testing.assert_allclose(sample["context"][0, :2], uncond_bits, rtol=0, atol=0)

    sample_b = dataset.sample(index=1)
    assert sample_b["context"].shape[-1] == 8


def test_cached_dataset_requires_records():
    import pytest

    with pytest.raises(ValueError, match="records must not be empty"):
        CachedDataset([], bucket_reso=(64, 96))
