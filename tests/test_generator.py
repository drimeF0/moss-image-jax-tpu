import numpy as np
import jax

from mossjax import generator as generator_mod, lora as lora_mod, smoke


class Tokenizer:
    def __call__(self, text):
        return {"input_ids": list(range(6))}


def _generator():
    dit, te, vae = smoke.build_params(0)
    return generator_mod.MossImageGenerator(
        dit, smoke.SMOKE_DIT, te, smoke.SMOKE_TE, vae, smoke.SMOKE_VAE, Tokenizer()
    )


def test_generate_returns_uint8_image():
    generator = _generator()
    image = generator.generate("cat", steps=2, width=32, height=32, seed=0, progress=False)
    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8


def test_low_mem_reloads_te_per_prompt():
    dit, te, vae = smoke.build_params(0)
    reloads = {"count": 0}

    def reload_fn():
        reloads["count"] += 1
        return te

    generator = generator_mod.MossImageGenerator(
        dit,
        smoke.SMOKE_DIT,
        None,
        smoke.SMOKE_TE,
        vae,
        smoke.SMOKE_VAE,
        Tokenizer(),
        te_reload_fn=reload_fn,
        free_te_after_encode=True,
    )
    generator.generate("one", steps=2, width=32, height=32, seed=0, progress=False)
    generator.generate("two", steps=2, width=32, height=32, seed=0, progress=False)
    assert reloads["count"] == 2
    assert generator._te_params is None


def test_scale_lora_multiplier():
    dit, te, vae = smoke.build_params(0)
    tree = lora_mod.init_lora(dit, smoke.SMOKE_DIT, rank=2, alpha=2.0, rng=jax.random.PRNGKey(0))
    scaled = generator_mod.scale_lora(tree, 2.5)
    first, second = tree["layers"][0]["proj"]["q"], scaled["layers"][0]["proj"]["q"]
    np.testing.assert_allclose(float(second["scale"]), float(first["scale"]) * 2.5)
    np.testing.assert_array_equal(np.asarray(second["A"]), np.asarray(first["A"]))
    assert generator_mod.scale_lora(None, 2.0) is None
    assert generator_mod.scale_lora(tree, 1.0) is tree


def test_generate_with_lora_multiplier_runs():
    dit, te, vae = smoke.build_params(0)
    tree = lora_mod.init_lora(dit, smoke.SMOKE_DIT, rank=2, alpha=2.0, rng=jax.random.PRNGKey(0))
    generator = generator_mod.MossImageGenerator(
        dit, smoke.SMOKE_DIT, te, smoke.SMOKE_TE, vae, smoke.SMOKE_VAE, Tokenizer(), lora=tree
    )
    image = generator.generate("cat", steps=2, width=32, height=32, seed=1, lora_multiplier=0.5, progress=False)
    assert image.shape == (32, 32, 3)
