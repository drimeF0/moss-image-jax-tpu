import jax.numpy as jnp
import numpy as np

from mossjax import sampling

from helpers import CFG, _params, _torch_model

EXPECTED_SIGMAS_50 = [
    1.0, 0.9959999918937683, 0.9919999837875366, 0.9869999885559082, 0.9829999804496765,
    0.9779999852180481, 0.9729999899864197, 0.9679999947547913, 0.9629999995231628, 0.9570000171661377,
    0.9520000219345093, 0.9459999799728394, 0.9390000104904175, 0.9329999685287476, 0.9259999990463257,
    0.918999969959259, 0.9120000004768372, 0.9049999713897705, 0.8970000147819519, 0.8880000114440918,
    0.8799999952316284, 0.8709999918937683, 0.8610000014305115, 0.8510000109672546, 0.8399999737739563,
    0.8289999961853027, 0.8169999718666077, 0.8050000071525574, 0.7910000085830688, 0.7770000100135803,
    0.7619999647140503, 0.7459999918937683, 0.7289999723434448, 0.7109999656677246, 0.6919999718666077,
    0.6710000038146973, 0.6480000019073486, 0.6229999661445618, 0.597000002861023, 0.5679999589920044,
    0.5359999537467957, 0.5009999871253967, 0.46300002932548523, 0.42100003361701965, 0.37300002574920654,
    0.32100000977516174, 0.26100000739097595, 0.19300000369548798, 0.11500000208616257, 0.024000000208616257,
    0.0,
]


def test_sampling_sigmas_match_upstream():
    sigmas = sampling.sampling_sigmas(50, 5.0)
    assert sigmas.dtype == np.float32
    assert len(sigmas) == len(EXPECTED_SIGMAS_50)
    np.testing.assert_allclose(sigmas, EXPECTED_SIGMAS_50, rtol=0, atol=1e-7)


def test_sampling_sigmas_are_descending():
    sigmas = sampling.sampling_sigmas(28, 5.0)
    assert sigmas[-1] == 0.0
    assert np.all(np.diff(sigmas) < 0)


def test_euler_step():
    latents = np.array([1.0], dtype=np.float32)
    prediction = np.array([2.0], dtype=np.float32)
    sigmas = np.array([0.5, 0.25], dtype=np.float32)
    out = sampling.euler_step(latents, prediction, sigmas, 0)
    np.testing.assert_allclose(out, [0.5], rtol=1e-6)


def test_generate_latents_smoke():
    params = _params(_torch_model())
    rng = np.random.default_rng(0)
    prompt = rng.standard_normal((1, 6, CFG.text_dim), dtype=np.float32)
    mask = np.ones((1, 6), dtype=np.int32)
    negative = np.zeros((1, 3, CFG.text_dim), dtype=np.float32)
    negative_mask = np.ones((1, 3), dtype=np.int32)

    out = sampling.generate_latents(
        params,
        CFG,
        prompt,
        mask,
        negative,
        negative_mask,
        height=32,
        width=32,
        steps=6,
        guidance_scale=3.0,
        flow_shift=5.0,
        uncond_skip_layers=(1,),
        seed=0,
        remat=True,
    )
    assert out.shape == (1, CFG.latent_channels, 2, 2)
    assert np.isfinite(out).all()

    no_cfg = sampling.generate_latents(
        params, CFG, prompt, mask, height=32, width=32, steps=6, guidance_scale=1.0, seed=0, remat=True
    )
    assert np.isfinite(no_cfg).all()
    assert not np.allclose(out, no_cfg, atol=1e-4)

    same = sampling.generate_latents(
        params,
        CFG,
        prompt,
        mask,
        negative,
        negative_mask,
        height=32,
        width=32,
        steps=6,
        guidance_scale=3.0,
        uncond_skip_layers=(1,),
        seed=0,
        remat=True,
    )
    np.testing.assert_array_equal(out, same)


def test_generate_latents_with_injected_initial_latents():
    params = _params(_torch_model())
    prompt = np.zeros((1, 4, CFG.text_dim), dtype=np.float32)
    mask = np.ones((1, 4), dtype=np.int32)
    initial = np.zeros((1, CFG.latent_channels, 2, 2), dtype=np.float32)
    out = sampling.generate_latents(
        params, CFG, prompt, mask, height=32, width=32, steps=4, guidance_scale=1.0, initial_latents=initial, remat=False
    )
    assert np.isfinite(out).all()


def test_generate_latents_progress(capsys):
    params = _params(_torch_model())
    prompt = np.zeros((1, 4, CFG.text_dim), dtype=np.float32)
    mask = np.ones((1, 4), dtype=np.int32)
    sampling.generate_latents(
        params, CFG, prompt, mask, height=32, width=32, steps=3, guidance_scale=1.0, seed=0
    )
    out = capsys.readouterr().out
    assert "[sample] denoising" in out
    assert "[sample] step 3/3" in out
    assert "eta" in out

    sampling.generate_latents(
        params, CFG, prompt, mask, height=32, width=32, steps=2, guidance_scale=1.0, seed=0, progress=False
    )
    assert capsys.readouterr().out == ""


def test_generate_latents_with_query_chunking():
    params = _params(_torch_model())
    prompt = np.zeros((1, 4, CFG.text_dim), dtype=np.float32)
    mask = np.ones((1, 4), dtype=np.int32)
    out = sampling.generate_latents(
        params,
        CFG,
        prompt,
        mask,
        height=32,
        width=32,
        steps=2,
        guidance_scale=1.0,
        seed=0,
        initial_latents=np.zeros((1, CFG.latent_channels, 2, 2), np.float32),
        query_chunk_size=2,
        progress=False,
    )
    assert np.isfinite(out).all()
