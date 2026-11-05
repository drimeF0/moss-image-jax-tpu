import math

import jax.numpy as jnp
import numpy as np

from mossjax import losses


def test_flow_scheduler_matches_torch_scheduler():
    timesteps, sigmas = losses.flow_scheduler(1000, shift=3.0)
    assert timesteps.shape == (1000,) and sigmas.shape == (1000,)
    assert timesteps.dtype == np.float32 and sigmas.dtype == np.float32
    expected_ts = {0: 1000.0, 1: 999.6663818359375, 500: 750.0, 999: 2.9940121173858643}
    expected_sig = {0: 1.0, 1: 0.9996663928031921, 500: 0.75, 999: 0.002994012087583542}
    for index, value in expected_ts.items():
        assert float(timesteps[index]) == value
    for index, value in expected_sig.items():
        assert float(sigmas[index]) == value
    unshifted, _ = losses.flow_scheduler(1000, shift=1.0)
    np.testing.assert_allclose(unshifted, np.linspace(1000, 1, 1000, dtype=np.float32), atol=0)


def test_truncated_logit_normal_matches_upstream():
    p = np.array([0.05, 0.4, 0.75, 0.99], dtype=np.float32)
    use = np.array([0.2, 0.8, 0.5, 0.1], dtype=np.float32)
    indices = np.array([7, 111, 500, 999], dtype=np.int64)
    got = losses.sample_truncated_logit_normal_indices(4, 1000, 0.0, 1.0, 1.0, 0.25, uniforms=(p, use, indices))
    assert got.tolist() == [7, 436, 662, 999]
    got = losses.sample_truncated_logit_normal_indices(4, 1000, 0.0, 1.0, 1.0, 0.0, uniforms=(p, use, indices))
    assert got.tolist() == [161, 436, 662, 911]
    offset = np.array([0.3, -0.2, 0.0, 0.5], dtype=np.float32)
    got = losses.sample_truncated_logit_normal_indices(4, 1000, -0.5, 0.8, 0.9, 0.5, offset, uniforms=(p, use, indices))
    assert got.tolist() == [7, 218, 458, 999]


def test_truncated_logit_normal_properties():
    rng = np.random.default_rng(0)
    indices = losses.sample_truncated_logit_normal_indices(4096, 1000, 0.0, 1.0, 1.0, 0.0, rng=rng)
    assert indices.min() >= 0 and indices.max() <= 999
    assert indices.dtype == np.int64
    with np.testing.assert_raises(ValueError):
        losses.sample_truncated_logit_normal_indices(4, 1000, 0.0, 1.0, 1.5, 0.0)


def test_density_sampling_matches_upstream():
    normal = np.array([0.5, -1.0, 2.0, -0.25], dtype=np.float32)
    rand = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
    u = losses.compute_density_for_timestep_sampling("logit_normal", 4, 0.5, 1.0, 1.29, uniforms=(normal, rand))
    np.testing.assert_allclose(u, [0.622459352, 0.268941432, 0.880797029, 0.437823504], rtol=0, atol=1e-7)
    offset = np.array([0.1, 0.2, -0.3, 0.0], dtype=np.float32)
    u = losses.compute_density_for_timestep_sampling(
        "logit_normal", 4, 0.5, 1.0, 1.29, offset, uniforms=(normal, rand)
    )
    np.testing.assert_allclose(u, [0.645656288, 0.310025513, 0.845534801, 0.437823504], rtol=0, atol=1e-7)
    u = losses.compute_density_for_timestep_sampling("mode", 4, 0.5, 1.0, 1.29, uniforms=(normal, rand))
    np.testing.assert_allclose(u, [0.802568495, 0.5, 0.197431475, 0.578878522], rtol=0, atol=1e-7)


def test_compute_loss_weighting():
    sigmas = np.array([0.5, 0.25], dtype=np.float32)
    np.testing.assert_allclose(losses.compute_loss_weighting("sigma_sqrt", sigmas), [4.0, 16.0], rtol=1e-6)
    expected = 2.0 / (math.pi * (1.0 - 2.0 * sigmas + 2.0 * sigmas**2))
    np.testing.assert_allclose(losses.compute_loss_weighting("cosmap", sigmas), expected, rtol=1e-6)
    np.testing.assert_array_equal(losses.compute_loss_weighting("logit_normal", sigmas), np.ones(2, np.float32))


def test_noisy_model_input_and_target():
    latents = np.zeros((2, 1, 1, 1), dtype=np.float32)
    noise = np.ones((2, 1, 1, 1), dtype=np.float32)
    sigmas = np.array([0.25, 0.75], dtype=np.float32)
    noisy = losses.noisy_model_input(jnp.asarray(latents), jnp.asarray(noise), jnp.asarray(sigmas))
    np.testing.assert_allclose(np.asarray(noisy).reshape(-1), [0.25, 0.75], rtol=1e-6)
    target = np.asarray(noise) - np.asarray(latents)
    np.testing.assert_array_equal(target.reshape(-1), [1.0, 1.0])


def test_rectified_flow_loss_matches_manual():
    rng = np.random.default_rng(0)
    prediction = rng.standard_normal((2, 3, 4, 5), dtype=np.float32)
    latents = rng.standard_normal((2, 3, 4, 5), dtype=np.float32)
    noise = rng.standard_normal((2, 3, 4, 5), dtype=np.float32)
    sigmas = np.array([0.2, 0.7], dtype=np.float32)
    weighting = losses.compute_loss_weighting("sigma_sqrt", sigmas)
    got = float(
        losses.rectified_flow_loss(
            jnp.asarray(prediction),
            jnp.asarray(latents),
            jnp.asarray(noise),
            jnp.asarray(sigmas),
            jnp.asarray(weighting),
        )
    )
    target = noise - latents
    elementwise = (prediction - target) ** 2 * weighting[:, None, None, None]
    expected = float(elementwise.mean(axis=(1, 2, 3)).mean())
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_rectified_flow_loss_with_sample_weights():
    prediction = np.zeros((2, 4), dtype=np.float32)
    latents = np.zeros((2, 4), dtype=np.float32)
    noise = np.ones((2, 4), dtype=np.float32)
    sigmas = np.array([0.5, 0.5], dtype=np.float32)
    loss = losses.rectified_flow_loss(
        jnp.asarray(prediction), jnp.asarray(latents), jnp.asarray(noise), jnp.asarray(sigmas), None, loss_weights=np.array([1.0, 3.0])
    )
    np.testing.assert_allclose(float(loss), (1.0 + 3.0) / 2.0, rtol=1e-6)
