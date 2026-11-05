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

