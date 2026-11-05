# Vendored reference: kohya-ss/sd-scripts

- Repo: https://github.com/kohya-ss/sd-scripts
- Branch: `main`
- Commit: `cb70b96858477e716a2096292ab8aab433277f38` (v0.12.3, 2026-10-26)
- License: Apache-2.0 (see `LICENSE.md`)

Files are copied verbatim from the pinned commit. **Do not edit them.** They are
the numerical reference for parity tests and the source of truth for checkpoint
formats, cache formats and LoRA file layout. To bump: re-download the tarball,
replace files, re-run `JAX_PLATFORMS=cpu pytest -q`, and update this file.

Vendored paths:

- `library/moss_image_models.py` — DiT, text-refiner, RoPE, attention glue
- `library/moss_image_utils.py` — checkpoint loaders, key mapping, tokenizer, TE
- `library/moss_image_train_utils.py` — sampler, sigma schedules, sampling
- `library/strategy_moss_image.py` — tokenize/TE-cache/latent-cache strategies
- `library/strategy_base.py`, `library/attention.py`, `library/utils.py`,
  `library/custom_offloading_utils.py`, `library/safetensors_utils.py`,
  `library/lora_utils.py`, `library/convrot_int8_utils.py`,
  `library/convrot_int8_kernels.py`, `library/flux2_vae.py`
- `moss_image_train_network.py`, `moss_image_minimal_inference.py`
- `tests/test_moss_image_*.py`, `tests/test_strategy_moss_image.py`
- `configs/moss_image_qwen35_9b/config.json`
- `docs/moss_image_train_network.md`

Tests import this tree via `sys.path` (see `tests/conftest.py`); the package name
`library` resolves to `tests/reference/kohya_sd_scripts/library`.
