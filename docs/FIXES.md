# Численные конвенции порта (FIXES)

Выстраданные конвенции upstream, которые обязаны сохраняться 1:1. При
сомнении — сверяться с `tests/reference/`, а не «улучшать». Каждая конвенция
покрыта тестом.

### FIXES #1. RMSNorm
Нормировка считается в float32 (`xf = x.float()`), затем результат кастится
обратно в dtype входа и умножается на weight: `y.to(x.dtype) * weight`, eps=1e-6.
Не считать в bf16 и не менять порядок «cast → multiply».

### FIXES #2. Тайм-эмбеддинг
`t = sigma * 1000` (без деления на 1000). Частоты и фаза считаются в float64:
`freq = 10000 ** (-arange(freq_dim/2) / (freq_dim/2))`, `phase = t.double() * freq`,
`embedding = cat(cos, sin).float()`. MLP — в dtype весов, результат — во float32,
модуляции — `[B, 6, D]` и `[B, 2, D]` из `time`.

### FIXES #3. SITU-GLU
`(4 * tanh(gate / 4) * sigmoid(gate)) * (25 * tanh(hidden / 25))`, где
`hidden, gate = chunk(ffn_up_gate(y), 2, -1)`. Порядок ветвей и константы — важны.

### FIXES #4. apply_rope
`rope` — это `[sin | cos]` (сначала sin). `sin`/`cos` дублируются
(`cat((s, s), -1)`), затем half-split ротация: `x * cos + cat((-b, a), -1) * sin`,
где `a, b = x.chunk(2, -1)`. RoPE применяется только к q/k, до attention.

### FIXES #5. RoPE-координаты и сетка частот
`num_bands = head_dim // 8`; bands строятся на CPU в fp32:
`1 / base ** (arange(num_bands) / num_bands)`. Координаты токенов:
текст `(0, j, 0, 0)`, image `(0, L, h - (H-1)/2, w - (W-1)/2)`, где `L` — **число
валидных** текстовых токенов сэмпла (не padded-длина). Выход —
`[sin(r,f,h,w) | cos(r,f,h,w)]` по `num_bands` на ось.

### FIXES #6. Маска внимания
Валидные токены — префикс `[image (N) | valid text]`; паддинг только справа.
Маскируется key-сторона (скоры паддинг-ключей = -inf); кванты-запросы паддинга
не существуют. Каждый сэмпл имеет ≥1 валидный текстовый токен. Содержимое
паддинг-позиций не влияет на выходы (тест padding-invariance).

### FIXES #7. dtypes по умолчанию
TE — bf16, VAE — fp32 независимо от mixed_precision. DiT в обучении — bf16,
паритет-тесты — fp32. Финальные слои кастятся в dtype веса
(`final_norm_video`, `final_linear_video`), text-refiner proj — в dtype веса.

### FIXES #8. Формы
DiT вход `[B, 128, H/16, W/16]` (patchified normalized latents), выход такой же.
N = H/16·W/16; последовательность `[image, text]`; image-upstream: H,W — сетка
токенов (`height=H/16` в RoPE). Паддинг латентов не существует (бакеты кратны 16).

### FIXES #9. Qwen3.5 TE
- `Qwen3_5RMSNorm` zero-centered: `x_fp32 * rsqrt(mean(x²)+eps)`, затем
  `(* (1 + weight))` в fp32 и каст к dtype входа (weight инициализируется
  нулями, используется `1 + w`).
- `Qwen3_5RMSNormGated` (GDN-выход): нормировка в fp32, `weight * x.to(dtype)`,
  затем `* silu(gate.float())` и каст обратно.
- Gated DeltaNet: q/k — l2norm + `1/sqrt(head_dim)`; `g = -exp(A_log) *
  softplus(a + dt_bias)` (softplus с threshold 20); `beta = sigmoid(b)`;
  рекуррентное состояние fp32 `[B, num_v_heads, k_dim, v_dim]`; q/k
  расширяются repeat_interleave до num_v_heads ДО l2norm-правила.
- GDN causal conv (kernel 4, depthwise, silu) с левым паддингом `k-1`.
- Full attention: `q_proj` даёт 2×head_dim (q + gate), gate применяется
  сигмоидой к выходу внимания до `o_proj`; q/k-norm zero-centered; GQA
  `repeat_kv` (kv-major, затем n_rep); partial RoPE (rotary_dim = 0.25·head_dim)
  — half-split на первых rotary_dim осях.
- mRoPE для текста: позиции всех трёх осей равны, поэтому
  `recomposition_frequencies` — тождество, cos/sin = `cat((cos(f), cos(f)))`.
  В upstream версия transformers зафиксирована 5.12.1; локальный паритет
  проверяется на установленной версии (5.22.0), см. PLAN §7.

### FIXES #10. Rectified flow (обучение)
- `noisy = (1 − σ)·x0 + σ·noise`, target `v = noise − x0`; `timesteps = σ·1000`
  (в DiT передаются как есть, без деления).
- σ-сэмплинг: `uniform`, `sigmoid` (`σ=sigmoid(scale·r)`, `r~N(0,1)`),
  `shift` (`σ` из sigmoid, затем `σ·shift/(1+(shift−1)σ)`), `flux_shift`
  (`time_shift(mu,1,σ)`, `mu = lin(256,0.5,4096,1.15)((h//2)(w//2))`),
  `truncated_logit_normal` (претрейн MOSS): `Z ~ N(logit_mean + offset, logit_std)`,
  `S = sigmoid(−Z)/cutoff`, индекс на убывающей сетке `T − ceil(S·T)`,
  с вероятностью `uniform_ratio` — равномерный индекс.
- Сетка планировщика: `u = linspace(1, T, T)[::-1]`, `σ = shift·u/(1+(shift−1)u)`,
  `timesteps = σ·T`.
- Лосс: поэлементный MSE (или huber), `mean` по всем осям кроме батча,
  умножение на per-sample weighting (для `logit_normal` = 1), затем `mean` по
  батчу. Реализация — `losses.rectified_flow_loss`.
- Оффсет применяется только при обучении (валидация без оффсета).

### FIXES #11. Формат кэшей (для полного TPU-пайплайна)
- Латенты: `<stem>_<W:04d>x<H:04d>_moss.npz`, ключ `latents_{H/8}x{W/8}`
  (raw [32, H/8, W/8], float32), плюс `original_size_*`, `crop_ltrb_*`,
  опционально `latents_flipped_*`, `alpha_mask_*`.
- TE-выходы: `<stem>_moss_te.npz`, ключи `num_candidates`, `prompt_embeds_{k}`
  (uint16 = bf16-биты, обрезано по валидной длине), `attn_mask_{k}`,
  `caption_{k}`, `uncond_embeds`/`uncond_mask`, `caption_dropout_rate`.
  Старый формат (0.12.1) — `prompt_embeds`/`attn_mask` без кандидатов.
- bf16-преобразование — round-to-nearest-even (проверено против torch,
  включая tie-случаи).

### FIXES #12. Сэмплинг (валидация/инференс)
- Сетка: `_sigma_grid(count, shift, total)` с `lower = shift/(total+shift−1)`;
  снап к убывающей тренировочной сетке (`training_shift=1`), удаление
  дубликатов, финальный 0. Для 50 шагов / shift 5.0 — 50 шагов (значения
  зафиксированы тестом).
- Euler: `x_{i+1} = x_i + (σ_{i+1} − σ_i)·v`, всё в fp32.
- `timestep = σ·1000` на каждом шаге; CFG: `uncond + scale·(pred − uncond)`;
  `uncond_skip_layers` — пропуск основных блоков в uncond-ветке (default 22).
- Размеры: `height/width` округляются вниз до кратных 16; шум — fp32.

### FIXES #13. .toml датасет (формат sd-scripts)
- Приоритет настроек: `[[datasets.subsets]]` > `[[datasets]]` >
  `[general]` (верхнеуровневые ключи вне `datasets` считаются general).
- Дефолты: `enable_bucket=false`, `min_bucket_reso=256`,
  `max_bucket_reso=1024`, `bucket_reso_steps=64`, `bucket_no_upscale=false`,
  `batch_size=1`, `num_repeats=1`, `caption_extension=".txt"`.
- `resolution`: int, `"W,H"` или `[W,H]`; bucketing считается от
  `original_size_*` в npz латентов через `BucketManager`; запись получает
  `bucket`, `num_repeats`, `caption_dropout_rate` (override кэша).
- Относительные пути (`image_dir`, `metadata_file`) разрешаются относительно
  каталога .toml (удобнее upstream-поведения от cwd; задокументировано).
- Сэмплирование записей — взвешенное по `num_repeats`; текст паддится до
  бакетов 128/256/512/1024/2048/4096.

### FIXES #14. Оптимизаторы
- `adamw` — как раньше (`b1=0.9`, `b2=0.999`, `eps=1e-8`, weight decay).
- `adafactor` — optax: `scale_parameter` ↔ `multiply_by_parameter_scale`,
  `relative_step` → `learning_rate=None`, плюс `decay_rate=0.8`,
  `clipping_threshold=1.0`, `momentum=None` (по умолчанию), `eps=1e-30`,
  `min_dim_size_to_factor=128`. Естественный масштаб lr у Adafactor другой
  (для tiny-тестов ~0.02 против AdamW ~0.05); clip-by-global-norm общий.

### FIXES #15. int8-загрузка на хосте и Colab-раннер
- `load_params(..., quantize_int8=True)` (DiT и TE) квантует веса в numpy ДО
  переноса на устройство: пик HBM на v5e-1 — только int8-дерево (DiT ~5 ГБ,
  TE ~9 ГБ), а не bf16+int8 одновременно. Схема квантования та же
  (per-output-channel, RNE, scale fp32 [out]).
- На точных полуцелых границах host- и device-квантование могут разойтись на
  1 уровень int8 (XLA-округление); обучение всегда идёт через host-путь,
  тест допускает ≤0.1% таких элементов.
- `scripts/colab_run.py` — единый TPU-процесс: стадии smoke/full, ошибки
  пишутся в `summary.json` без потери предыдущих стадий. Ноутбук:
  `notebooks/colab_tpu.ipynb`. Не инициализировать JAX в ячейках ноутбука —
  только через `!python`.

### FIXES #16. Precision на TPU: HIGHEST только для fp32
- `lax.Precision.HIGHEST` обязателен для fp32-матумов на TPU (иначе они
  эмулируются на bf16), но для bf16 его запрашивать нельзя: Mosaic падает
  `INTERNAL ... online-softmax: Bad lhs type` при `tpu.matmul` с
  `contract_precision<fp32>` (bf16×bf16→f32) в фьюжене внимания (v5e).
- Правило порта: `precision = HIGHEST if dtype == fp32 else DEFAULT` — в
  `model.linear/attention`, `quant.weight_only_linear`, `text_encoder._linear`
  и attention. bf16 MXU и так аккумулирует в fp32; VAE (fp32) остаётся на HIGHEST.
- Регрессионный тест: bf16-шаг обучения на tiny-модели + проверка выбора precision.

### FIXES #17. Пре-квантизованные чекпойнты (формат int8+scale)
- `scripts/prepare_quantized.py` (ядро — `mossjax.prepare`): один safetensors на
  модель — `<key>` int8 для целевых слоёв + `<key>.scale` fp32 [out], остальные
  веса как есть (bf16 сохраняется нативно, safetensors+ml_dtypes).
- Metadata: `mossjax_format="int8+scale"`, `mossjax_act_dtype` (bf16/fp32).
  Лоадеры (`weights.load_params`, `text_encoder.load_params`) авто-детектят
  `<key>.scale` и собирают quant-leaf'ы без пересчёта; `--int8` для таких
  файлов — no-op.
- Область квантования 1:1 с ConvRot: DiT `layers.*.proj.*`, TE проекции +
  `embed_tokens`. Готовить один раз на высокоRAM-машине (пик при подготовке:
  выходной dict + один входной шард), затем на v5e-1 грузить только int8.

### FIXES #18. Инференс на TPU: jit, порядок стадий, стриминговые лоадеры
- **Всё на инференсе должно быть jit**: `sampling.generate_latents` и
  `caching.encode_prompt_embeds` компилируют forward (params передаются
  аргументами, не запекаются). Eager-диспатчи 5B/9B давали «зависание» и
  активный churn памяти. Первый шаг сэмплинга включает XLA-компиляцию — это
  ожидаемо, прогресс печатается после каждого шага.
- Порядок стадий `colab_run --mode full`: VAE → TE (кэш + промпт сэмпла) →
  освобождение TE → DiT (обучение и сэмплинг на тех же параметрах, без
  повторной загрузки). TE (int8 ~9.5 ГБ) и DiT (int8 ~5 ГБ, плюс активации)
  не находятся в памяти одновременно.
- Стриминговые лоадеры: `load_params` в DiT/TE читает и конвертирует по одному
  тензору (`safe_open` → jnp → `del`), host-пик ≈ один тензор вместо всего
  чекпойнта. Скейлы prepared-файлов держатся отдельно (десятки МБ).
- Прогресс: баннеры стадий и времена загрузки, `[sample] step i/N … eta`.

### FIXES #19. Кэш-стадия: jit VAE/TE, бакеты длин, ресайз в bucket
- VAE-энкодер в кэше теперь jit (фабрика по cfg через `lru_cache`), иначе
  eager-диспатчи конвов давали ~80 с на картинку.
- TE-энкод — постоянный jit с **паддинг-бакетами длин** (128…4096):
  right-padding безопасен (GDN и full attention каузальны, выходы валидного
  префикса совпадают с непаддингом — тест padded==unpadded), а число
  компиляций 9B-модели ограничено числом бакетов, а не числом капшенов
  (раньше `jax.jit` создавался заново на каждый вызов → перекомпиляция).
- Латенты кэшируются ровно в bucket-резолюшн (`reso`), `original_size` —
  истинный размер исходной картинки; для режима cache-dir кэш режется в
  фиксированный `--width/--height`. Имя файла `_WxH_moss.npz`, ключ
  `latents_{H/8}x{W/8}` — согласовано с тренером/лоадером.
- Первая картинка печатает пометку «includes XLA compilation»: её время —
  это компиляции (VAE-энкодер несколько секунд, TE 9B — один раз на бакет
  длины), начиная со второй картинки время должно упасть до секунд.

### FIXES #20. Память при сэмплинге на 16 ГБ
- В `sample.py` TE освобождается сразу после кодирования промпта, DiT — перед
  загрузкой VAE/декодом; одновременно TE (9.5 ГБ) и DiT (5 ГБ) не живут
  (иначе OOM: forward на 1024² просит ~5.7 ГБ temp).
- `attention(..., query_chunk_size)`: scores [B,H,Lq,S] считаются чанками по
  запросам (softmax по-прежнему по всем ключам — результат идентичен, тест
  chunked==unchunked). Чанкинг сквозь `model.forward`/`generate_latents`;
  в `sample.py`/`colab_run` включается автоматически при числе image-токенов
  > 2048 (grid 1024²=4096) со значением 1024, либо вручную `--query-chunk`.

### FIXES #21. OOM в обучении на больших бакетах (16 ГБ)
- Симптом: датасет с бакетами до 1024², OOM после N шагов. Причина — не утечка,
  а пик/фрагментация HBM: каждый новый размер бакета = отдельная компиляция, а
  attention материализует [B,H,S,S] на самом большом бакете.
- `TrainConfig.query_chunk_size` (дефолт 1024): чанковый attention и в
  обучении (результат идентичен, тест); CLI `--query-chunk` /
  `--train-query-chunk`, 0 = выключить.
- Лог обучения теперь печатает реальный расход HBM:
  `mem 9.3/15.7 GB (peak 11.2)` (через `jax.local_devices()[0].memory_stats()`).
- Рекомендации для 16 ГБ: держать `max_bucket_reso` ≤ 768 в .toml (или учить
  в фиксированном `--resolution`), и следить, чтобы кэширование и обучение
  использовали один и тот же набор бакетов (иначе падение на fallback-ключах
  и лишние компиляции).











