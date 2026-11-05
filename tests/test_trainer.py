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
