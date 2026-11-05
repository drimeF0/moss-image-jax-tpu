import numpy as np
import torch

from mossjax.data import buckets, cache


def test_make_bucket_resolutions_matches_upstream():
    assert buckets.make_bucket_resolutions((1024, 1024), 256, 1024, 64) == [
        (256, 1024), (320, 1024), (384, 1024), (448, 1024), (512, 1024), (576, 1024), (640, 1024),
        (704, 1024), (768, 1024), (832, 1024), (896, 1024), (960, 1024), (1024, 256), (1024, 320),
        (1024, 384), (1024, 448), (1024, 512), (1024, 576), (1024, 640), (1024, 704), (1024, 768),
        (1024, 832), (1024, 896), (1024, 960), (1024, 1024),
    ]
    assert buckets.make_bucket_resolutions((512, 768), 256, 1024, 64) == [
        (256, 1024), (320, 1024), (384, 896), (384, 960), (384, 1024), (448, 832), (512, 704),
        (512, 768), (576, 576), (576, 640), (640, 576), (704, 512), (768, 512), (832, 448),
        (896, 384), (960, 384), (1024, 256), (1024, 320), (1024, 384),
    ]


def test_select_bucket_matches_upstream():
    cases = {
        (1024, 1024): ((1024, 1024), (1024, 1024), 0.0),
        (800, 600): ((1024, 768), (1024, 768), 0.0),
        (512, 768): ((704, 1024), (704, 1056), 0.02083333333333337),
        (2048, 1024): ((1024, 512), (1024, 512), 0.0),
        (300, 900): ((320, 1024), (341, 1024), -0.020833333333333315),
        (640, 480): ((1024, 768), (1024, 768), 0.0),
    }
    no_upscale_cases = {
        (1024, 1024): ((1024, 1024), (1024, 1024), 0.0),
        (800, 600): ((768, 576), (800, 600), 0.0),
        (512, 768): ((512, 768), (512, 768), 0.0),
        (2048, 1024): ((1408, 704), (1408, 704), 0.0),
        (300, 900): ((256, 896), (300, 900), -0.047619047619047616),
        (640, 480): ((640, 448), (640, 480), 0.09523809523809534),
    }
    for no_upscale, expected in ((False, cases), (True, no_upscale_cases)):
        manager = buckets.BucketManager(no_upscale, (1024, 1024), 256, 1024, 64)
        if not no_upscale:
            manager.make_buckets()
        for size, (reso, resized, ar_error) in expected.items():
            got = manager.select_bucket(*size)
            assert got[0] == reso, (no_upscale, size, got)
            assert got[1] == resized, (no_upscale, size, got)
            assert abs(got[2] - ar_error) < 1e-12, (no_upscale, size, got)


def test_bf16_bits_match_torch():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(4096).astype(np.float32) * 10.0).astype(np.float32)
    got = cache.float32_to_bf16_bits(x)
    expected_x = torch.from_numpy(x.copy()).to(torch.bfloat16).view(torch.uint16).numpy()
    np.testing.assert_array_equal(got, expected_x)

    ties = np.array([(bits << 16) | 0x8000 for bits in range(0x3F00, 0x3F40)], dtype=np.uint32).view(np.float32)
    got = cache.float32_to_bf16_bits(ties)
    expected_ties = torch.from_numpy(ties.copy()).to(torch.bfloat16).view(torch.uint16).numpy()
    np.testing.assert_array_equal(got, expected_ties)

    floats = cache.uint16_to_float32(got)
    assert floats.dtype == np.float32
    np.testing.assert_array_equal(cache.uint16_to_float32(cache.float32_to_bf16_bits(x)), cache.uint16_to_float32(expected_x))


def test_te_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    embeds0 = rng.standard_normal((5, 8), dtype=np.float32)
    embeds1 = rng.standard_normal((3, 8), dtype=np.float32)
    uncond = rng.standard_normal((4, 8), dtype=np.float32)
    path = tmp_path / "image_moss_te.npz"
    cache.save_te_outputs(path, [(embeds0, "caption zero"), (embeds1, "caption one")], uncond, 0.1)

    with np.load(path) as npz:
        assert set(npz.files) >= {
            "num_candidates", "prompt_embeds_0", "attn_mask_0", "caption_0",
            "prompt_embeds_1", "attn_mask_1", "caption_1", "uncond_embeds", "uncond_mask", "caption_dropout_rate",
        }
        assert npz["prompt_embeds_0"].dtype == np.uint16
        assert npz["num_candidates"] == 2

    record = cache.load_te_outputs(path)
    assert len(record["candidates"]) == 2
    np.testing.assert_array_equal(record["candidates"][0]["mask"], np.ones(5, dtype=np.int32))
    np.testing.assert_allclose(record["candidates"][0]["embeds"], cache.uint16_to_float32(cache.float32_to_bf16_bits(embeds0)))
    np.testing.assert_allclose(record["candidates"][1]["embeds"], cache.uint16_to_float32(cache.float32_to_bf16_bits(embeds1)))
    assert record["candidates"][0]["caption"] == "caption zero"
    np.testing.assert_allclose(record["uncond"]["embeds"], cache.uint16_to_float32(cache.float32_to_bf16_bits(uncond)))
    assert abs(record["caption_dropout_rate"] - 0.1) < 1e-6

    selected = cache.select_candidate(record, np.random.default_rng(2))
    assert selected["embeds"].shape[1] == 8


def test_te_cache_old_format(tmp_path):
    embeds = np.random.default_rng(3).standard_normal((6, 8), dtype=np.float32)
    path = tmp_path / "old_moss_te.npz"
    np.savez(
        path,
        prompt_embeds=cache.float32_to_bf16_bits(embeds),
        attn_mask=np.ones(6, dtype=np.int32),
    )
    record = cache.load_te_outputs(path)
    assert len(record["candidates"]) == 1
    assert record["uncond"] is None
    np.testing.assert_allclose(record["candidates"][0]["embeds"], cache.uint16_to_float32(cache.float32_to_bf16_bits(embeds)))


def test_latents_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    latents = rng.standard_normal((32, 12, 8), dtype=np.float32)
    path = cache.latents_npz_path(str(tmp_path / "image.png"), (64, 96))
    assert path.endswith("image_0064x0096_moss.npz")
    cache.save_latents(path, latents, original_size=[64, 96], crop_ltrb=[0, 0, 64, 96])

    with np.load(path) as npz:
        assert "latents_12x8" in npz.files
        assert "original_size_12x8" in npz.files

    np.testing.assert_array_equal(cache.load_latents(path, (64, 96)), latents)


def test_latents_cache_fallback_without_suffix(tmp_path):
    latents = np.random.default_rng(5).standard_normal((32, 12, 8), dtype=np.float32)
    path = tmp_path / "old_moss.npz"
    np.savez(path, latents=latents)
    np.testing.assert_array_equal(cache.load_latents(str(path), (64, 96)), latents)


def test_pad_embeds():
    a = np.random.default_rng(6).standard_normal((3, 4), dtype=np.float32)
    b = np.random.default_rng(7).standard_normal((5, 4), dtype=np.float32)
    embeds, mask = cache.pad_embeds([a, b])
    assert embeds.shape == (2, 5, 4) and mask.shape == (2, 5)
    np.testing.assert_array_equal(embeds[0, :3], a)
    np.testing.assert_array_equal(embeds[0, 3:], np.zeros((2, 4)))
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


def test_find_cache_pairs(tmp_path):
    import numpy as np

    save_te = tmp_path / "img_moss_te.npz"
    np.savez(save_te, prompt_embeds=np.zeros((1, 4), np.uint16), attn_mask=np.ones(1, np.int32))
    np.savez(tmp_path / "img_0064x0096_moss.npz", latents=np.zeros((32, 8, 6), np.float32))
    np.savez(tmp_path / "other_moss_te.npz", prompt_embeds=np.zeros((1, 4), np.uint16), attn_mask=np.ones(1, np.int32))
    records = cache.find_cache_pairs(str(tmp_path))
    assert len(records) == 1
    assert records[0]["key"].endswith("img")
    assert records[0]["latents"][0].endswith("img_0064x0096_moss.npz")
