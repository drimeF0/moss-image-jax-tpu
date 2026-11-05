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

### Ф0. Инфраструктура (сделано)
- Репо, pyproject, .gitignore, вендоринг reference `tests/reference/kohya_sd_scripts/`.
- PIN: upstream commit, лицензия.
- CPU-прогон `JAX_PLATFORMS=cpu pytest -q`.

### Ф1. Порт DiT forward + паритет (в работе)
- `src/mossjax/config.py` — MOSS_IMAGE_CONFIG.
- `src/mossjax/model.py` — чистые функции forward (RMSNorm, SITU-GLU,
  apply_rope, 4D-RoPE, блоки, text-refiner, модуляции, голова).
- `src/mossjax/weights.py` — конвертация torch state_dict ↔ JAX params
  (strip префиксов, split fused qkv, drop `rope.bands`).
- Тесты: tiny-конфиг vs vendored torch `MossImage` (fp32, maxdiff ≤ 1e-6),
  инвариантность к содержимому паддинга, skip_layers.

### Ф2. Чекпойнт и int8 базовой модели
- Реальные safetensors: diffusers shards (первый шард), ComfyUI single-file.
- int8 per-output-channel symmetric + scale [out]; reverse-тест точности
  (bf16 vs int8, cos > 0.9999 на tiny), проверка HLO `s8×s8→s32` на TPU.
- Хранение: int8-веса + fp32 scale в pytree; native-слои в bf16/fp32.

### Ф3. Qwen3.5-9B TE (24 Gated DeltaNet + 8 full attn)
- Порт forward, per-sample, без паддинга; паритет с
  `transformers.Qwen3_5TextModel` (CPU, fp32/bf16).
- int8 для проекций; `mamba_ssm_dtype float32` для рекурренции.
- Отдельный режим: encoder-pass c кэшированием TE-выходов (`*_moss_te.npz`
  формат), стриминг весов с host RAM для 16 ГБ.
- Токенизация: чат-темплейт, `MAX_TOKEN_LENGTH=4096`, кандидаты-капшены.

### Ф4. FLUX.2 VAE
- Port encode/decode (+mid-attention), patchify/normalize;
  паритет с vendored `flux2_vae.py` / diffusers `AutoencoderKLFlux2` (fp32).
- Кэш латентов `*_moss.npz`; decode для валидации (при наличии бюджета).

### Ф5. Данные
- Bucket manager (кратно 16), `.toml` датасета, `enable_multiline_caption`.
- Чтение/запись npz-кэшей upstream 1:1; padding/text-length как трассируемые
  скаляры + маски, статические бакеты форм (без рекомпиляций).
- caption dropout + выбор кандидата per-step.

### Ф6. LoRA и совместимость
- JAX LoRA (A/B, scale=alpha/r), regex-выбор модулей, per-module dim/lr,
  LoRA+ (опционально).
- Экспорт в kohya-формат и импорт kohya-файлов; roundtrip-тест с upstream
  `networks.lora_generic` на CPU (загрузка/merge/сверка 1e-6).
- Опционально: save/load TE-LoRA — вне первого этапа.

### Ф7. Тренер
- Rectified-flow loss, t-sampling (sigma/uniform/sigmoid/shift/flux_shift/
  truncated_logit_normal), offset/weighting — по
  `library/moss_image_train_utils.py`.
- optax AdamW/Adafactor, grad clip, grad accumulation, LR scheduler, EMA?
- `jax.checkpoint` (remat) на блоки; int8 base + bf16 LoRA; masked loss.
- Валидация: euler + CFG + skip layers; лог в upstream-формате.
- Чекпойнты: safetensors LoRA (для интеропа) + orbax resume.

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
