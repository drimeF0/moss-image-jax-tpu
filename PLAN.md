# План порта MOSS-Image LoRA-обучения на TPU (JAX, int8)

Документ для агентов, продолжающих работу. Описывает цель, зафиксированные
решения, дорожную карту по фазам и критерии приёмки. Обновлять при изменении
решений.

## 1. Контекст

Источник: [`kohya-ss/sd-scripts`](https://github.com/kohya-ss/sd-scripts),
ветка **main**, pinned commit
`cb70b96858477e716a2096292ab8aab433277f38` (v0.12.3, 2026-10-26). На main уже
есть скрипты MOSS-Image (`moss_image_train_network.py`,
`library/moss_image_{models,utils,train_utils}.py`,
`library/strategy_moss_image.py`, `library/flux2_vae.py`); edit-вариант
(`exp-moss-edit`) вне scope.

Ссылочный код зафиксирован в `tests/reference/kohya_sd_scripts/` (Apache-2.0,
см. LICENSE.md). Импортируется в тестах для паритета; менять его нельзя.

Что обучается: LoRA (`networks.lora_generic`, regex-выбор модулей) на 5B
single-stream DiT. Qwen3.5-9B текстовый энкодер и FLUX.2 VAE заморожены.

## 2. Зафиксированные решения

1. **Железо: Colab TPU v5e-1, 1 чип, 16 ГБ HBM.** Sharding-код пишем, но
   single-chip — основной режим; коллективов в горячем пути нет. Модель в
   **int8** (frozen base) + LoRA в bf16.
2. **Полный TPU-пайплайн.** На TPU работают: Qwen3.5-9B TE (int8), FLUX.2 VAE
   (fp32), DiT (int8), обучение, валидация/сэмплинг. PyTorch на TPU не
   используется; torch — только тестовый референс на CPU. Токенизация
   (Qwen3.5 tokenizer, ~20 МБ) — обычный CPU-код (`tokenizers`/`transformers`).
3. **Полная совместимость LoRA-файлов с upstream обязательна.** Формат
   `networks/lora_generic.py`: ключи `lora_unet_*`, `lora_down/up.weight`,
   `alpha`, scale = alpha/rank, init down=kaiming_uniform(a=sqrt(5)),
   up=zeros. Roundtrip-тест: JAX save → загрузка и merge в upstream torch
   (CPU) → битовая/1e-6 сверка; и обратно.
4. **int8 считаем сами** из официальных bf16/fp32 safetensors (per-output-channel
   symmetric), а не через CUDA-кернелы ConvRot. Чтение ComfyUI ConvRot-файлов —
   опционально (dequant→requant), не default. Квантизуем только FLOP-доминантные
   Linear (q/k/v/o, ffn_up_gate/ffn_down, qkv/proj/ffn text-refiner, TE-проекции);
   эмбеддеры/нормы/модуляции/головы — native dtype.
5. **DiT хранит веса в torch-layout `[out, in]`** (x @ W.T) — чекпойнт-маппинг
   без транспонирования, простой экспорт/roundtrip. На инференсе можно
   транспонировать в память один раз. Для int8 это же естественная раскладка
   (scale по out-каналам).
6. **Численные конвенции upstream сохраняются 1:1** (см. docs/FIXES.md), при
   сомнении сверяться с `tests/reference/`, а не «улучшать».
7. Репозиторий: `moss-image-jax-tpu`, пакет `src/mossjax`, тесты `tests/`.
   CPU-CI обязателен: `JAX_PLATFORMS=cpu pytest -q`.

## 3. Ключевые факты о моделях

### DiT (MOSS-Image, 5B)
- `MOSS_IMAGE_CONFIG`: dim 3072, ffn 8192, heads 24 (head_dim 128), layers 42,
  text_dim 4096, freq_dim 256, latent_channels 128 (32ch × patchify 2×2),
  rope_base 10000.
- Последовательность `[image (N), text (L)]`; текст right-padded, валидные
  токены — префикс; image-токены всегда валидны. N = (H/16)·(W/16).
- Shared-модуляция 6·D на все блоки + head-модуляция 2·D, из `t = σ·1000`.
- RMSNorm: нормировка в fp32, weight в dtype входа. SITU-GLU:
  `4·tanh(g/4)·σ(g) · 25·tanh(h/25)`.
- Attention: qk-norm per head, RoPE на q/k, выход [B,L,H·D]. Маска — только на
  key-стороне (padded text).
- RoPE 4D `(r,f,h,w)`: текст `(0, j, 0, 0)`; image `(0, L, h−(H−1)/2, w−(W−1)/2)`;
  `num_bands = head_dim/8 = 16`, выход `[sin(4 оси), cos(4 оси)]`;
  `apply_rope` — half-split (`cat(-b,a)`), sin повторяется дважды.
- Время: синусоида в float64, затем fp32; MLP → fp32; модуляции — из
  `time` в dtype соответствующего Linear.
- Финальный выход: `final_norm_video` (dtype веса) → head-модуляция →
  `final_linear_video` → [B, 128, H, W].

### Qwen3.5-9B TE (frozen, bf16/int8)
- 32 слоя: 24 × `linear_attention` (Gated DeltaNet: key 16×128, value 32×128,
  conv4, gated) + 8 × `full_attention` (16 heads × 256, 4 KV-heads, partial
  RoPE 0.25, mrope_section [11,11,10]). `mamba_ssm_dtype: float32`.
- Кодируем по одному примеру без паддинга (Gated DeltaNet чувствителен к
  паддингу). Выход — last hidden state 4096d.
- Конфиг: `configs/moss_image_qwen35_9b/config.json`, transformers 5.12.1
  (`Qwen3_5TextModel`); vision tower не загружается.

### FLUX.2 VAE (frozen, fp32)
- 32-канальный латент, downscale 8, «small decoder». `patchify_latents`:
  `z.view(B,C,H/2,2,W/2,2).permute(0,1,3,5,2,4)` → 128 каналов;
  `normalize_latents` mean/var (128). Референс: `library/flux2_vae.py`,
  diffusers `AutoencoderKLFlux2`.

### Сэмплер (валидация)
- Euler: sigmas из training-grid с flow_shift, `t = σ·1000`; default 50 шагов,
  CFG 3.0, `uncond_skip_layers=22` (skip основных блоков для uncond-ветки).

### Кэши upstream
- Латент: `*_moss.npz` (raw [32, H/8, W/8]).
- TE-выходы: `*_moss_te.npz` (uint16-биты bf16, `prompt_embeds_k`,
  `attn_mask_k`, `caption_k`, `num_candidates`, uncond-эмбеддинги).

## 4. Бюджет памяти v5e-1 (16 ГБ)

| Компонент | Размер | Режим |
|---|---|---|
| DiT int8 | ~4.9 ГБ | резидент на время train |
| DiT bf16-часть (эмбеддеры/нормы/головы) | ~0.2 ГБ | резидент |
| LoRA + optax | десятки МБ | резидент |
| Активации (N+L ≤ 8192, remat) | ~2–4 ГБ | пик шага |
| Qwen3.5-9B int8 | ~9.3 ГБ | **не резидент**: отдельный pass (кэш TE) или стриминг по слоям |
| FLUX.2 VAE fp32 | ~1–2 ГБ | отдельный pass (encode latents / decode samples) |

Следствие: стадийный пайплайн `encode TE → encode VAE → train`; TE и DiT
никогда не в памяти одновременно. При `--cache_text_encoder_outputs` /
`--cache_latents` (upstream-эквивалент) это естественно.

## 5. Фазы

Статус: Ф0 ✅ · Ф1 ✅ · Ф2 ⏳ (int8 DiT+TE, лоадер реальных шардов ждёт файлов) ·
Ф3 ✅ (forward+int8+кэш TE; стриминг весов с host RAM — дальше) ·
Ф4 ✅ · Ф5 ✅ (бакеты/кэши/датасет; .toml — дальше) · Ф6 ✅ ·
Ф7 ✅ (лосс/шаг/E2E-цикл/чекпойнты LoRA/сэмплер/grad accumulation/CLI; orbax-resume и лог-совместимость — дальше) ·
Ф8 — запланирована.

### Ф0. Инфраструктура (сделано)
- Репо, pyproject, .gitignore, вендоринг reference `tests/reference/kohya_sd_scripts/`.
- PIN: upstream commit, лицензия.
- CPU-прогон `JAX_PLATFORMS=cpu pytest -q`.

### Ф1. Порт DiT forward + паритет (сделано)
- `src/mossjax/config.py` — MOSS_IMAGE_CONFIG.
- `src/mossjax/model.py` — чистые функции forward (RMSNorm, SITU-GLU,
  apply_rope, 4D-RoPE, блоки, text-refiner, модуляции, голова).
- `src/mossjax/weights.py` — конвертация torch state_dict ↔ JAX params
  (strip префиксов, split fused qkv, drop `rope.bands`).
- Тесты: tiny-конфиг vs vendored torch `MossImage` (fp32, факт maxdiff 2.4e-7),
  инвариантность к содержимому паддинга, skip_layers, roundtrip весов.

### Ф2. Чекпойнт и int8 базовой модели (сделано частично)
- int8 per-output-channel symmetric (`quant.py`): storage `int8` + `scale` [out],
  weight-only-деquant linear (autodiff-safe) и native int8 MXU путь
  (`dot_general(..., preferred_element_type=int32)`) для frozen inference.
  Область — `layers.*.proj.*`, как upstream ConvRot (`DIT_CONVROT_INT8_TARGET_KEYS=["layers."]`).
- `weights.state_dict_to_params` принимает официальные (diffusers-шарды) и
  ComfyUI-ключи (fused qkv), префиксы `model.diffusion_model.`.
- Осталось: smoke на реальном чекпойнте (первый шард), HLO-проверка s8×s8→s32
  на v5e, custom_vjp int8-fwd/bf16-bwd (перф, не память).

### Ф3. Qwen3.5-9B TE (готов, кроме стриминга)
- `src/mossjax/text_encoder.py`: embeddings, zero-centered RMSNorm,
  GatedDeltaNet (conv1d + recurrent gated delta rule, fp32 state), full attention
  (GQA, partial RoPE, q-gate), MLP, `config_from_hf`/`load_config`.
- Паритет с `transformers.Qwen3_5TextModel` (eager, CPU): полный forward и
  отдельные linear/full слои, вход без паддинга, per-sample.
- int8 (`quant.quantize_text_encoder_params`): проекции + `embed_tokens`,
  как upstream (in_proj_a/b — native); cos > 0.9999 к fp32.
- Кэш: `caching.write_te_cache` → `*_moss_te.npz` 1:1 с upstream.
- Осталось: стриминг весов TE с host RAM для 16 ГБ (по слоям),
  опционально chunked-вариант delta rule.

### Ф4. FLUX.2 VAE (сделано)
- `src/mossjax/vae.py`: NCHW-порт encoder/decoder (GroupNorm, Resnet, Attention,
  Downsample/Upsample), patchify/normalize, загрузчик BFL/ComfyUI и
  diffusers-конвертер.
- Паритет: encode 1e-5, decode 5e-5 (fp32, tiny-конфиг), diffusers-конвертация
  бит-в-бит, roundtrip safetensors.

### Ф5. Данные (сделано ядро)
- `src/mossjax/data/buckets.py` — `make_bucket_resolutions` и `BucketManager`
  (оба режима: с апскейлом и `no_upscale`), паритет с upstream по значениям.
- `src/mossjax/data/cache.py` — чтение/запись `*_moss.npz` и
  `*_moss_te.npz` ровно в формате upstream (включая старый TE-формат),
  bf16 RNE биты, `pad_embeds`, выбор кандидата, поиск пар кэшей.
- Осталось: парсер `.toml` датасета (subsets, metadata.json, caption dropout,
  multiline), итератор бакетов, flip/alpha-маска.

### Ф6. LoRA и совместимость (сделано)
- `src/mossjax/lora.py`: kohya-раскладка `lora_unet_layers_N_proj_*`
  (`lora_down.weight`, `lora_up.weight`, `alpha`), init/export/import/merge,
  файловый I/O (`save_lora_file` с metadata/`load_lora_file`).
- `model.forward(..., lora=...)`: LoRA-ветка bf16 поверх int8-базы.
- Roundtrip: наш safetensors грузится в upstream `networks.lora_generic`
  (strict), значения/alpha совпадают, форвард патченного torch == JAX (1e-5).
- Осталось: per-module `network_reg_dims`/`network_reg_lrs`, LoRA+.

### Ф7. Тренер (ядро сделано)
- `src/mossjax/losses.py`: все режимы t-sampling (включая
  `truncated_logit_normal` с паритетом к upstream), сетка планировщика,
  weighting, rectified-flow лосс (reduction 1:1).
- `src/mossjax/trainer.py`: optax AdamW + clip, `train_step` (grad только по
  LoRA, база заморожена), remat блоков в `model.forward`.
- `src/mossjax/train.py`: `train_loop` по `CachedDataset` (кандидаты капшенов,
  caption dropout, паддинг текста до бакетов 128..4096), лог loss/current,
  чекпойнты LoRA каждые N шагов + финальный файл.
- `caching.py`: VAE-кэш латентов и TE-кэш (train-time нормализация из
  `bn.running_*`).
- `src/mossjax/sampling.py`: Euler-сэмплер с CFG и skip-layers, сетка сигм
  1:1 с upstream (значения зафиксированы тестом).
- `trainer.make_accumulated_train_step`: накопление градиентов (тест:
  сумм. градиент == полный батч).
- `scripts/train.py`: CLI (реальные safetensors DiT/VAE, int8, кэши, LoRA-выход).
- E2E-тест: картинки → VAE-кэш → TE-кэш → обучение → LoRA-файл грузится в
  upstream `networks.lora_generic` (strict) и форвардит.
- Лоадеры реальных чекпойнтов: `weights.load_params` (префиксы/fused qkv),
  `text_encoder.load_params` (skip vision/lm_head), `vae.load_vae_params`.
- Осталось: orbax/resume, лог по upstream-формату, `.toml`-датасет,
  валидационный цикл в train_loop, прогон на реальных весах v5e-1.
### Ф8. TPU v5e-1 и приёмка
- Smoke на реальном чекпойнте int8; memory_analysis бюджета; attention:
  explicit/splash для S до 8192.
- Бенч: first step vs steady, ms/step, tokens/s; отчёт в docs/.
- Colab-ноутбук с одним TPU-процессом (Colab-правила: инициализация TPU
  только в раннере).

## 6. Чек-лист приёмки

1. `JAX_PLATFORMS=cpu pytest -q` зелёный (паритет + юниты).
2. TPU smoke: DiT forward, шаг LoRA, сэмплинг на реальных формах.
3. HLO-подтверждение: int8 s8×s8→s32; отсутствие f32→bf16 эмуляции в паритете.
4. LoRA roundtrip с upstream torch: 1e-6 (merge), загрузка файлов в обе стороны.
5. int8 vs bf16: A/B по лоссу/сэмплам; cos > 0.9999.
6. Полный пайплайн на TPU без torch: TE-кэш → VAE-кэш → train → sample.
7. Чекпойнт/resume на mesh; лог, читаемый upstream-парсером.
8. Бенч v5e-1: compile отдельно, steady ms/step, память.

## 7. Риски

1. **Gated DeltaNet TE** — самый дорогой порт (рекуррентный scan, conv4).
   Митигация: паритет по слоям, отдельные тесты на 1 слой.
2. **16 ГБ потолок**: TE не резидент с DiT; стриминг весов TE с host RAM;
   remat всех блоков; аккуратность с пиками активаций (S до 8192).
3. **int8-расхождения**: только веса (weight-only) в v1 обучения; активации
   bf16; A/B против bf16 обязателен.
4. **Полная LoRA-совместимость**: точный формат ключей генерика и scale;
   roundtrip-тесты в CI (torch CPU).
5. **Splash-attention**: ограничения Pallas по dtype/маскам; fallback —
   explicit на бакетах ≤ 8192.
6. **transformers API drift** (5.12.1 в upstream): для паритета TE
   зафиксировать версию в pyproject тестов.
7. Ветка main может измениться — обновление только сознательным «bump PIN»
   с повторным вендорингом и прогоном паритета.

## 8. Репозиторий и запуск

```bash
pip install -e .[test]
JAX_PLATFORMS=cpu pytest -q            # CPU-паритет
python scripts/train_toy.py            # позже: tiny-тренировка
```

Структура:

```
src/mossjax/          # пакет порта (model, weights, quant, lora, trainer, data/, text_encoder/, vae/)
tests/                # pytest, включая паритет с tests/reference
tests/reference/      # vendored upstream + PIN.md
docs/FIXES.md         # численные конвенции, выстраданные грабли
scripts/              # convert/bench/train runners
```
