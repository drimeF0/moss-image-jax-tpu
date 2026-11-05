# moss-image-jax-tpu

Порт LoRA-обучения [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) из
[`kohya-ss/sd-scripts`](https://github.com/kohya-ss/sd-scripts) (ветка `main`,
коммит `cb70b96858477e716a2096292ab8aab433277f38`, v0.12.3) на JAX/Flax + TPU.

- Целевое железо: Colab TPU v5e-1 (1 чип, 16 ГБ HBM), базовая модель в int8.
- Полный TPU-пайплайн: токенизация (CPU), Qwen3.5-9B TE, FLUX.2 VAE, DiT,
  обучение LoRA, валидация/сэмплинг — без PyTorch на TPU.
- Обязательная полная совместимость LoRA-файлов с upstream
  (`networks/lora_generic.py`): сохранённый safetensors загружается в
  sd-scripts и наоборот.
- Edit-вариант (`moss_image_train_network_edit.py`) вне scope.

План порта и текущий статус: [PLAN.md](PLAN.md).
Численные конвенции (обязательны к сохранению): [docs/FIXES.md](docs/FIXES.md).

## Статус

- Готово: DiT forward (паритет fp32 2.4e-7), int8-база (weight-only/MXU),
  FLUX.2 VAE (encode/decode/patchify), LoRA с полной совместимостью файлов
  (roundtrip через upstream `networks.lora_generic`).
- Готов весь CPU-контур: кэши (VAE/TE), датасет, обучение LoRA, экспорт, интероп.
- Дальше: стриминг весов TE для 16 ГБ, .toml-датасет, валидация/сэмплинг, прогон на Colab v5e-1.
- Ссылочный torch-код зафиксирован в `tests/reference/kohya_sd_scripts/`.

## Быстрый старт

Colab/Kaggle TPU (один TPU-процесс на сессию): ноутбук
[`notebooks/colab_tpu.ipynb`](notebooks/colab_tpu.ipynb) — сначала smoke-тест
без скачивания весов, затем реальный прогон:

```bash
python scripts/prepare_quantized.py --dit ... --text-encoder ... --output-dir ...  # один раз
python scripts/colab_run.py --mode smoke --output-dir /content/moss_out/smoke
python scripts/colab_run.py --mode full --dit ..._dit_int8.safetensors \
    --vae ... --text-encoder ..._te_int8.safetensors --tokenizer ... \
    --images-dir ... --prompts prompts.json --output-dir ...
```

`prepare_quantized.py` сохраняет int8-веса и скейлы в один safetensors (загрузка —
обычное чтение, без квантования на каждом запуске).

Отдельные стадии: `scripts/make_caches.py`, `scripts/train.py`
(AdamW/Adafactor, `--dataset-config dataset.toml`), `scripts/sample.py`.

## Тесты

```bash
JAX_PLATFORMS=cpu pytest -q
```
