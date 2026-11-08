import jax
import jax.numpy as jnp
import numpy as np
import torch

from mossjax import lora, model as jm, quant, trainer

from helpers import CFG, _params, _torch_model


def _batch(batch=2, height=3, width=5, length=7, seed=0):
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((batch, CFG.latent_channels, height, width), dtype=np.float32)
    noise = rng.standard_normal(latents.shape, dtype=np.float32)
    sigmas = rng.random(batch, dtype=np.float32)
    context = rng.standard_normal((batch, length, CFG.text_dim), dtype=np.float32)
    mask = np.ones((batch, length), dtype=np.int32)
    mask[0, 4:] = 0
    weighting = np.ones(batch, dtype=np.float32)
    inputs = tuple(jnp.asarray(x) for x in (latents, noise, sigmas, context, mask, weighting))
    return inputs


def _run_steps(params, steps=20, learning_rate=5e-2, remat=True):
    config = trainer.TrainConfig(rank=4, learning_rate=learning_rate, remat=remat)
    state, optimizer = trainer.init_train_state(params, CFG, config, jax.random.PRNGKey(0))
    step = trainer.make_train_step(CFG, config, optimizer)
    batch = _batch()
    losses = []
    for _ in range(steps):
        state, metrics = step(state, params, *batch)
        losses.append(float(metrics["loss"]))
    return losses


def test_train_step_reduces_loss():
    params = _params(_torch_model())
    losses = _run_steps(params)
    assert np.mean(losses[-5:]) < 0.5 * np.mean(losses[:5])
    assert all(np.isfinite(losses))


def test_train_step_with_quantized_base_reduces_loss():
    params = quant.quantize_params(_params(_torch_model()), CFG)
    losses = _run_steps(params)
    assert np.mean(losses[-5:]) < 0.5 * np.mean(losses[:5])


def test_remat_matches_plain_forward():
    params = _params(_torch_model())
    lora_params = lora.init_lora(params, CFG, rank=2, alpha=2.0, rng=jax.random.PRNGKey(1))
    batch = _batch()
    outputs = []
    for remat in (False, True):
        outputs.append(
            np.asarray(
                jm.forward(params, batch[0], batch[2], batch[3], batch[4], CFG, lora=lora_params, remat=remat)
            )
        )
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-6, rtol=0)


def test_train_step_changes_only_lora():
    params_before = _params(_torch_model())
    params = jax.tree.map(lambda leaf: jnp.asarray(leaf), params_before)
    config = trainer.TrainConfig(rank=2, learning_rate=5e-2)
    state, optimizer = trainer.init_train_state(params, CFG, config, jax.random.PRNGKey(2))
    step = trainer.make_train_step(CFG, config, optimizer)
    before = jax.tree.map(lambda leaf: np.array(leaf), state["lora"])
    state, _ = step(state, params, *_batch())
    after = jax.tree.map(lambda leaf: np.array(leaf), state["lora"])
    changed = [not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after))]
    assert any(changed)
    flat_params = jax.tree.leaves(params)
    assert all(isinstance(leaf, jax.Array) for leaf in flat_params)


def test_lora_file_roundtrip(tmp_path):
    params = _params(_torch_model())
    tree = lora.init_lora(params, CFG, rank=2, alpha=2.0, rng=jax.random.PRNGKey(3))
    path = tmp_path / "moss_lora.safetensors"
    lora.save_lora_file(path, tree, CFG, dim=2, alpha=2.0, metadata={"ss_learning_rate": 1e-4})
    back = lora.load_lora_file(path, params, CFG)
    first = jax.tree.leaves(tree)
    second = jax.tree.leaves(back)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(np.array(a), np.array(b))

    from safetensors import safe_open

    with safe_open(str(path), framework="np") as handle:
        metadata = handle.metadata()
    assert metadata["ss_network_module"] == "networks.lora_generic"
    assert metadata["ss_network_dim"] == "2"
    assert metadata["ss_learning_rate"] == "0.0001"


def test_init_train_state_rejects_nothing():
    params = _params(_torch_model())
    state, optimizer = trainer.init_train_state(params, CFG, trainer.TrainConfig(rank=4), jax.random.PRNGKey(0))
    assert state["step"] == 0
    leaves = jax.tree.leaves(state["lora"])
    assert len(leaves) == CFG.num_layers * 6 * 3
    assert optimizer is not None


def _micro_batch(seed, batch=1, height=3, width=5, length=7):
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((batch, CFG.latent_channels, height, width), dtype=np.float32)
    noise = rng.standard_normal(latents.shape, dtype=np.float32)
    sigmas = rng.random(batch, dtype=np.float32)
    context = rng.standard_normal((batch, length, CFG.text_dim), dtype=np.float32)
    mask = np.ones((batch, length), dtype=np.int32)
    weighting = np.ones(batch, dtype=np.float32)
    return latents, noise, sigmas, context, mask, weighting


def test_accumulated_grads_match_full_batch():
    from mossjax import losses

    params = _params(_torch_model())
    state, optimizer = trainer.init_train_state(params, CFG, trainer.TrainConfig(rank=2), jax.random.PRNGKey(5))
    lora0 = state["lora"]

    def loss_for(lora, batch):
        latents, noise, sigmas, context, mask, weighting = (jnp.asarray(x) for x in batch)
        noisy = losses.noisy_model_input(latents, noise, sigmas)
        prediction = jm.forward(params, noisy, sigmas * 1000, context, mask, CFG, lora=lora, remat=True)
        return losses.rectified_flow_loss(prediction, latents, noise, sigmas, weighting)

    micro0 = _micro_batch(10)
    micro1 = _micro_batch(11)
    full = tuple(np.concatenate([a, b], axis=0) for a, b in zip(micro0, micro1))

    _, grads_full = jax.value_and_grad(loss_for)(lora0, full)
    _, grads0 = jax.value_and_grad(loss_for)(lora0, micro0)
    _, grads1 = jax.value_and_grad(loss_for)(lora0, micro1)
    grads_mean = jax.tree.map(lambda a, b: 0.5 * (a + b), grads0, grads1)
    for a, b in zip(jax.tree.leaves(grads_full), jax.tree.leaves(grads_mean)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)

    stacked = tuple(jnp.asarray(np.stack([np.asarray(a), np.asarray(b)])) for a, b in zip(micro0, micro1))
    step = trainer.make_accumulated_train_step(
        CFG, trainer.TrainConfig(rank=2, learning_rate=5e-2), optimizer, accumulation_steps=2
    )
    state2, metrics = step(state, params, stacked)
    expected_loss = float(loss_for(lora0, full))
    np.testing.assert_allclose(float(metrics["loss"]), expected_loss, rtol=1e-5)
    assert not np.array_equal(np.asarray(state2["lora"]["layers"][0]["proj"]["q"]["A"]), np.asarray(lora0["layers"][0]["proj"]["q"]["A"]))


def test_adafactor_train_step_reduces_loss():
    params = _params(_torch_model())
    config = trainer.TrainConfig(
        rank=4,
        learning_rate=0.02,
        optimizer="adafactor",
        adafactor_scale_parameter=False,
        adafactor_relative_step=False,
        adafactor_clipping_threshold=1.0,
        remat=True,
    )
    state, optimizer = trainer.init_train_state(params, CFG, config, jax.random.PRNGKey(7))
    step = trainer.make_train_step(CFG, config, optimizer)
    batch = _batch(seed=5)
    losses = []
    for _ in range(25):
        state, metrics = step(state, params, *batch)
        losses.append(float(metrics["loss"]))
    assert np.mean(losses[-5:]) < 0.5 * np.mean(losses[:5])
    assert all(np.isfinite(losses))


def test_adafactor_relative_step_runs():
    params = _params(_torch_model())
    config = trainer.TrainConfig(
        rank=2,
        optimizer="adafactor",
        adafactor_relative_step=True,
        adafactor_scale_parameter=True,
    )
    state, optimizer = trainer.init_train_state(params, CFG, config, jax.random.PRNGKey(8))
    step = trainer.make_train_step(CFG, config, optimizer)
    state, metrics = step(state, params, *_batch(seed=6))
    assert np.isfinite(float(metrics["loss"]))


def test_make_optimizer_rejects_unknown():
    with np.testing.assert_raises(ValueError):
        trainer.make_optimizer(trainer.TrainConfig(optimizer="sgd"))


def test_bf16_precision_selection():
    from jax import lax

    assert jm._precision_for(jnp.float32) == lax.Precision.HIGHEST
    assert jm._precision_for(jnp.bfloat16) == lax.Precision.DEFAULT
    assert jm._precision_for(jnp.zeros((2,), jnp.float32)) == lax.Precision.HIGHEST
    assert jm._precision_for(jnp.zeros((2,), jnp.bfloat16)) == lax.Precision.DEFAULT
    from mossjax import quant

    assert quant.matmul_precision(jnp.float32) == lax.Precision.HIGHEST
    assert quant.matmul_precision(jnp.bfloat16) == lax.Precision.DEFAULT


def test_bf16_training_step_runs():
    params = jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
        _params(_torch_model()),
    )
    config = trainer.TrainConfig(rank=2, learning_rate=1e-2, remat=True)
    state, optimizer = trainer.init_train_state(params, CFG, config, jax.random.PRNGKey(9))
    step = trainer.make_train_step(CFG, config, optimizer)
    batch = _batch(seed=7)
    for _ in range(2):
        state, metrics = step(state, params, *batch)
    assert np.isfinite(float(metrics["loss"]))


def test_query_chunking_preserves_loss():
    from mossjax import losses

    params = _params(_torch_model())
    batch = _batch(seed=8)
    latents, noise, sigmas, context, mask, weighting = (jnp.asarray(x) for x in batch)

    def loss_for(chunk):
        noisy = losses.noisy_model_input(latents, noise, sigmas)
        prediction = jm.forward(
            params, noisy, sigmas * 1000, context, mask, CFG, remat=True, query_chunk_size=chunk
        )
        return losses.rectified_flow_loss(prediction, latents, noise, sigmas, weighting)

    full = float(loss_for(None))
    chunked = float(loss_for(2))
    np.testing.assert_allclose(full, chunked, rtol=1e-5, atol=1e-6)


def test_init_train_state_with_initial_lora():
    from mossjax import lora

    params = _params(_torch_model())
    tree = lora.init_lora(params, CFG, rank=2, alpha=2.0, rng=jax.random.PRNGKey(0))
    state, _ = trainer.init_train_state(
        params, CFG, trainer.TrainConfig(rank=4), jax.random.PRNGKey(1), initial_lora=tree
    )
    for a, b in zip(jax.tree.leaves(tree), jax.tree.leaves(state["lora"])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    rank, alpha = lora.lora_rank_alpha(tree)
    assert rank == 2 and abs(alpha - 2.0) < 1e-6
    rank, alpha = lora.lora_rank_alpha({}, default_rank=5, default_alpha=7.0)
    assert rank == 5 and alpha == 7.0
