# Layer-hot decoding: results

Machine: MacBook Air M4 (Mac16,12), 16 GB, fanless, on the charger. Every number below has an
`env.json` next to it in `results/<timestamp>/`. Rounds that tripped a discard rule (not on
charger, CPU idle < 90 % before the round, thermal limit < 100, swap growth, free disk < 5 GB)
are excluded from medians; the benches wait for a quiet machine before each round (Photos'
`mediaanalysisd` wakes up regularly on AC power).

Software: Python 3.12.11, mlx 0.32.4, mlx-lm 0.32.2, llama.cpp `e3483e4` (2026-11-14 master,
includes the few-row Metal kernels of PR #29869, merged 2026-10-05), dflash-mlx 0.1.8.
Models: `mlx-community/Qwen3-{0.6B,1.7B,4B}-4bit` (MLX), `unsloth/Qwen3-*-GGUF` Q4_0 (llama.cpp),
drafters `z-lab/Qwen3-4B-DFlash-b16` (DFlash) and `mgoin/Qwen3-4B-speculator.dflash2` (DFlash2).

Tables are printed by `python -m bench.report` from the JSON in `results/`.

## Step 1: c(K), the cost of K rows in one decode sweep

### MLX (`bench/row_cost.py`)

| model | ctx | sweep ms (K=1) | c(2) | c(3) | c(4) | c(8) | LM head share | 1 layer, 1 row (ms) | 16k head (ms) | source |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.6B | 512 | 5.4 | 1.11 | 1.85 | 1.65 | 2.78 | 20% | 0.44 | 0.35 | `20261114-192334-row_cost_mlx` |
| 0.6B | 2048 | 7.3 | 1.11 | 1.56 | 1.63 | 3.12 | 15% | 0.52 | 0.35 | `20261114-192334-row_cost_mlx` |
| 1.7B | 512 | 12.8 | 1.02 | 1.28 | 1.59 | 2.93 | 15% | 1.01 | 0.44 | `20261114-192334-row_cost_mlx` |
| 1.7B | 2048 | 14.7 | 0.97 | 1.29 | 1.60 | 2.72 | 13% | 0.77 | 0.42 | `20261114-192334-row_cost_mlx` |
| 4B | 512 | 29.5 | 1.04 | 1.26 | 1.52 | 2.76 | 8% | 1.29 | 0.56 | `20261114-221454-row_cost_mlx_4B` |
| 4B | 2048 | 31.6 | 0.96 | 1.32 | 1.63 | 2.89 | 8% | 1.37 | 0.50 | `20261114-221454-row_cost_mlx_4B` |

### llama.cpp Metal and CPU (`bench/row_cost_llama.py`, `llama-batched-bench -pps`, 128 decode steps)

| model | backend | ctx | step ms (K=1) | serial tok/s | c(2) | c(3) | c(4) | c(8) |
|---|---|---|---|---|---|---|---|---|
| 0.6B | cpu | 512 | 9.3 | 107.7 | 1.49 | 2.08 | 1.73 | 3.39 |
| 0.6B | cpu | 2048 | 26.4 | 37.9 | 1.20 | 1.64 | 1.75 | 3.45 |
| 0.6B | metal | 512 | 6.3 | 158.7 | 1.50 | 1.62 | 1.73 | 2.16 |
| 0.6B | metal | 2048 | 8.6 | 116.1 | 1.57 | 1.91 | 2.26 | 3.61 |
| 1.7B | metal | 512 | 12.9 | 77.7 | 1.46 | 1.48 | 1.53 | 1.75 |
| 1.7B | metal | 2048 | 15.1 | 66.1 | 1.48 | 1.67 | 1.87 | 2.67 |
| 4B | metal | 512 | 27.5 | 36.4 | 1.38 | 1.44 | 1.50 | 1.77 |
| 4B | metal | 2048 | 32.0 | 31.2 | 1.51 | 1.77 | 2.04 | 3.15 |

llama-bench tg128 at depth 512 (cross-check of the K=1 column): 0.6B/metal 175.9 tok/s, 1.7B/metal 80.7 tok/s, 4B/metal 38.2 tok/s, 0.6B/cpu 116.5 tok/s

On llama.cpp the K=1 column runs 4-10 % below `llama-bench` tg128 at the same depth (0.6B 159 vs
176, 1.7B 78 vs 81, 4B 36 vs 38 tok/s): batched-bench runs several sequences in a unified KV
cache, which costs a little even at one sequence. The CPU run (`-ngl 0`, 0.6B, 4 threads) tests
the literal "layer stays in cache" claim: c(2) = 1.49 at ctx 512, so a second row is not cheap on
the CPU either; the CPU path is not purely weight-bandwidth bound at this size.

**Reading.** On MLX a second row is close to free once the model is not tiny: c(2) = 1.11 (0.6B),
1.02 (1.7B), 1.04 (4B). MLX keeps its matrix-vector kernel for 2 rows; from 3 rows on the cost
grows roughly linearly (c(4) ≈ 1.5-1.65, c(8) ≈ 2.7-3.1). llama.cpp is the other way round: the
step from 1 to 2 rows is the expensive one (it leaves the mat-vec kernel for the few-row kernels),
and from there to 8 rows the curve is flat. **Gate (c(2) > 1.6 everywhere): not tripped.** The
hardware half of the idea holds: a second token rides the weight sweep for 2-11 % extra on MLX.

The LM head is 20 % / 15 % / 8 % of a sweep (0.6B / 1.7B / 4B). A reduced 16k-row head costs
0.35-0.56 ms. A one-layer, one-row call costs 0.44-1.3 ms, most of it host round trip on the
small models (a 0.6B layer is ~0.16 ms of GPU time inside a full sweep).

## Step 2: a(E), shallow-exit agreement with the full model (frozen model)

Data: ~118k positions per model of the model's own greedy output on 475 prompts (150 chat,
175 code, 150 wikitext prose openings), 256 new tokens each. Split by topic group (a chat topic,
a code task or a prose paragraph is entirely train or entirely test), 80/20. The first, leakier
split (by prompt) inflated E=2 top-1 on 0.6B from 32 % to 38 %; all numbers below use the group split.

Heads: logit lens (no training); affine map + frozen reduced-vocab head (16k most frequent output
tokens, rows taken from the model's quantized head) + abstain logit; step 4a adapter (below).
"Predicted speedup" replays the step 3 loop over the held-out sequences (chained guesses, stop at
the first abstain or low-confidence guess, accept the matching prefix + 1 bonus token) and prices
each cycle with the step 1 timings at ctx 512, including the measured draft-head and one-layer call
times. "No host overhead" subtracts the measured per-call round trip from every draft call, as a
fused graph (e.g. a llama.cpp port) would.

**0.6B** (`20261114-215117-acceptance_0.6B`): 94050 train / 21636 held-out positions. Sanity: logit lens at the last layer = full model 100.0%; teacher-forced label = generated token 99.1%. Next-token classes: content 39%, function 30%, punct/ws 25%, subword 6%.

| E | head | top-1 | punct/ws | function | subword | content | precision @ coverage (τ=0.5) | best replay (kmax, τ) | tokens/sweep | cost/sweep | predicted speedup | same, no host overhead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lens | 0.3% | 0% | 1% | 1% | 0% | 1% @ 25% | (2, 0.95) | 1.001 | 1.317 | **0.760** | 0.825 |
| 1 | affine | 29.1% | 46% | 39% | 23% | 12% | 51% @ 39% | (2, 0.3) | 1.237 | 1.304 | **0.948** | 1.056 |
| 1 | adapter | 34.6% | 54% | 46% | 21% | 16% | 59% @ 47% | (2, 0.3) | 1.299 | 1.335 | **0.973** | 1.082 |
| 2 | lens | 0.6% | 0% | 1% | 2% | 0% | 1% @ 22% | (2, 0.95) | 1.000 | 1.385 | **0.722** | 0.780 |
| 2 | affine | 32.0% | 50% | 43% | 25% | 13% | 52% @ 46% | (2, 0.5) | 1.227 | 1.363 | **0.900** | 0.990 |
| 2 | adapter | 36.0% | 57% | 47% | 21% | 17% | 59% @ 48% | (2, 0.5) | 1.264 | 1.409 | **0.897** | 0.984 |
| 4 | lens | 1.4% | 1% | 3% | 3% | 1% | 2% @ 17% | (2, 0.95) | 1.001 | 1.357 | **0.738** | 0.798 |
| 4 | affine | 38.0% | 55% | 53% | 30% | 17% | 57% @ 55% | (2, 0.5) | 1.296 | 1.465 | **0.885** | 0.969 |
| 4 | adapter | 38.5% | 58% | 54% | 28% | 16% | 63% @ 47% | (2, 0.5) | 1.278 | 1.398 | **0.914** | 1.003 |
| 8 | lens | 2.4% | 2% | 4% | 4% | 1% | 4% @ 20% | (2, 0.95) | 1.002 | 1.412 | **0.710** | 0.765 |
| 8 | affine | 40.1% | 59% | 55% | 28% | 19% | 58% @ 60% | (2, 0.7) | 1.267 | 1.493 | **0.848** | 0.923 |
| 8 | adapter | 40.4% | 60% | 55% | 27% | 19% | 65% @ 51% | (2, 0.5) | 1.308 | 1.533 | **0.853** | 0.929 |

Best: E=1 adapter, predicted 0.973x → gate: **step 4a/4b first**.

**1.7B** (`20261114-200238-acceptance_1.7B`): 96238 train / 21950 held-out positions. Sanity: logit lens at the last layer = full model 100.0%; teacher-forced label = generated token 99.3%. Next-token classes: content 39%, function 30%, punct/ws 26%, subword 5%.

| E | head | top-1 | punct/ws | function | subword | content | precision @ coverage (τ=0.5) | best replay (kmax, τ) | tokens/sweep | cost/sweep | predicted speedup | same, no host overhead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lens | 0.1% | 0% | 0% | 1% | 0% | 0% @ 24% | (2, 0.95) | 1.000 | 1.215 | **0.823** | 0.897 |
| 1 | affine | 27.9% | 44% | 39% | 13% | 11% | 45% @ 43% | (2, 0.3) | 1.239 | 1.177 | **1.052** | 1.171 |
| 2 | lens | 0.1% | 0% | 0% | 1% | 0% | 0% @ 14% | (2, 0.95) | 1.000 | 1.205 | **0.830** | 0.898 |
| 2 | affine | 31.3% | 50% | 43% | 14% | 12% | 48% @ 52% | (2, 0.3) | 1.278 | 1.207 | **1.059** | 1.177 |
| 4 | lens | 0.3% | 0% | 1% | 1% | 0% | 0% @ 9% | (2, 0.95) | 1.000 | 1.215 | **0.823** | 0.896 |
| 4 | affine | 35.8% | 56% | 49% | 19% | 14% | 52% @ 58% | (2, 0.5) | 1.286 | 1.233 | **1.044** | 1.162 |
| 8 | lens | 1.3% | 1% | 2% | 2% | 1% | 3% @ 9% | (2, 0.95) | 1.000 | 1.223 | **0.818** | 0.889 |
| 8 | affine | 37.5% | 58% | 52% | 16% | 16% | 52% @ 64% | (2, 0.7) | 1.265 | 1.257 | **1.006** | 1.105 |

Best: E=2 affine, predicted 1.059x → gate: **step 4a/4b first**.

**4B** (`20261114-225257-acceptance_4B`): 98456 train / 22728 held-out positions. Sanity: logit lens at the last layer = full model 100.0%; teacher-forced label = generated token 99.4%. Next-token classes: content 40%, function 29%, punct/ws 25%, subword 6%.

| E | head | top-1 | punct/ws | function | subword | content | precision @ coverage (τ=0.5) | best replay (kmax, τ) | tokens/sweep | cost/sweep | predicted speedup | same, no host overhead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lens | 0.3% | 0% | 0% | 5% | 0% | 0% @ 46% | (2, 0.95) | 1.000 | 1.097 | **0.912** | 0.947 |
| 1 | affine | 27.1% | 41% | 40% | 14% | 11% | 47% @ 39% | (2, 0.3) | 1.220 | 1.087 | **1.123** | 1.177 |
| 2 | lens | 0.0% | 0% | 0% | 0% | 0% | 0% @ 28% | (2, 0.95) | 1.000 | 1.094 | **0.914** | 0.948 |
| 2 | affine | 30.0% | 48% | 42% | 15% | 12% | 46% @ 51% | (2, 0.3) | 1.259 | 1.130 | **1.114** | 1.168 |
| 4 | lens | 0.0% | 0% | 0% | 0% | 0% | 0% @ 13% | (2, 0.95) | 1.000 | 1.086 | **0.921** | 0.954 |
| 4 | affine | 34.3% | 55% | 48% | 17% | 14% | 51% @ 58% | (2, 0.5) | 1.271 | 1.126 | **1.129** | 1.180 |
| 8 | lens | 0.2% | 0% | 0% | 1% | 0% | 1% @ 5% | (2, 0.95) | 1.000 | 1.092 | **0.916** | 0.949 |
| 8 | affine | 36.3% | 57% | 51% | 18% | 15% | 50% @ 64% | (2, 0.7) | 1.246 | 1.149 | **1.085** | 1.133 |

Best: E=4 affine, predicted 1.129x → gate: **step 4a/4b first**.

**Reading.**
- The zero-training logit lens is useless at shallow exits (0.1-2.4 %): after 1-8 layers the
  residual stream still mostly encodes the current token, not the next one.
- A trained affine exit agrees with the full model 28-40 % of the time, rising slowly with depth.
  It is a grammar-token predictor, as expected: 44-60 % on punctuation/whitespace and 39-55 % on
  function words, 11-19 % on content words.
- Deeper exits do not pay: the extra acceptance (a few points) is smaller than the extra draft cost.
  kmax = 2 (one guess per sweep) wins everywhere; chaining a second guess rarely survives.
- **Gate (≥ 1.2x predicted): not passed on any model.** Best: 0.97x (0.6B, adapter E=1),
  1.06x (1.7B, E=2), 1.13x (4B, E=4; E=1 gives 1.12x). Without per-call host overhead: 1.08x,
  1.18x, 1.18x. The larger the model, the smaller the fixed draft cost relative to a sweep, so the
  prediction rises with size, but acceptance does not (27-36 % at E ≤ 8 on all three models).
- Interpretation: one guess per sweep at ~30 % acceptance is worth ~1.25 tokens per sweep at best;
  with a free second row and zero draft cost that caps the frozen-model idea near 1.2-1.25x.

## Step 3: lossless loop (`layerhot/generate.py`, `tests/test_lossless.py`)

Built and verified even though the step 2 gate did not pass, because steps 4 and 5 run on it.
Two draft modes: `step` (host sync after each guess, stop at the first unconfident one) and
`chain` (all kmax-1 guesses drafted on device with one sync, unconfident tail trimmed).

**Correctness** (`bench/lossless_run.py`, 24 held-out prompts x 128 tokens, compared with
`mlx_lm` `generate_step` greedy):

| model | head, E, kmax, τ, mode | tokens/sweep | prompts that diverge | tie distance at the divergence (bf16 steps: count) | result | dir |
|---|---|---|---|---|---|---|
| 4B | affine, 1, 2, 0.3, step | 1.227 | 13 / 24 | 0: 7, 1: 6 | pass | `20261115-083219-lossless_4B_affine_E1_k2_step` |
| 1.7B | affine, 2, 2, 0.3, step | 1.294 | 15 / 24 | 0: 8, 1: 6, 2: 1 | pass | `20261115-083553-lossless_1.7B_affine_E2_k2_step` |
| 0.6B | adapter, 1, 4, 0.3, chain | 1.429 | 19 / 24 | 0: 6, 1: 12, 2: 1 | pass | `20261115-083736-lossless_0.6B_adapter_E1_k4_chain` |

Every divergence happens where the full model's top two logits are tied or one bf16 step apart
(at most two), measured along mlx-lm's own path (prefill, then one token at a time). The verify
pass computes several rows at once and writes their KV entries with slightly different rounding
than one-row decoding; at an exact or one-step tie that flips the argmax. With kmax = 1 (no guessing) the
loop matched mlx-lm token for token in the one check made (0.6B, 64 tokens). A first version of the test measured margins with a single prefill of
prompt + prefix; that is a different computation (up to 0.45 logits off on 4B) and wrongly flagged
one 4B divergence as real; inspected and fixed (`tests/test_lossless.py`). dflash-mlx documents
the same effect for its own verify ("MLX dispatch divergence"), and diverges on 5 of 6 prompts
in step 5.

**Speed** (`bench/ab_ours.py`, `20261115-082716-ab_ours_4B`): Qwen3-4B, E = 1, kmax = 2, τ = 0.3,
same process, serial / ours / ours / serial per prompt after a 90 s warm-up, 256 tokens.

| | code (3 prompts) | prose (3 prompts) | all |
|---|---|---|---|
| ours / mlx-lm greedy | 0.89x | 0.82x | **0.86x** |
| tokens per sweep | 1.19-1.29 | 1.13-1.17 | |

Where the time goes (4B, ~200-token context): a synchronous one-row step takes 29.7 ms; mlx-lm's
`generate_step` emits a token every 26.1 ms because it builds step t+1 while the GPU runs step t
(`async_eval`). The loop cannot overlap that way: it needs the guess on the host to decide the
rows, and the accept count on the host to trim the KV cache, so each cycle has two round trips. A
cycle costs ~36 ms = 1.21 synchronous sweeps (the step 2 cost model predicted 1.09; the rest is
Python dispatch of the split graph: two embeddings, two layer-range calls, the trim) and yields
1.19 tokens, i.e. 30.4 ms per token. So the step 2 predictions were optimistic twice over: they
priced the baseline as a synchronous sweep, and they missed ~11 % of dispatch cost.

## Step 4: raising acceptance (model frozen, still lossless)

### 4a. Kangaroo-style adapter (`layerhot/adapter.py`)

One causal self-attention block on the layer-E states (its own KV cache, initialised from the
model's next attention layer), then the frozen final norm and reduced head. Trained 6 epochs with
cross-entropy on the full model's token + KL to the full model's distribution. Rows "adapter" in
the step 2 tables.

Measured on 0.6B at E = 1, 2, 4, 8 (rows "adapter" in the 0.6B table above):

| E | affine top-1 | adapter top-1 | affine predicted | adapter predicted | adapter, no host overhead |
|---|---|---|---|---|---|
| 1 | 29.1 % | 34.6 % | 0.948 | 0.973 | 1.082 |
| 2 | 32.0 % | 36.0 % | 0.900 | 0.897 | 0.984 |
| 4 | 38.0 % | 38.5 % | 0.885 | 0.914 | 1.003 |
| 8 | 40.1 % | 40.4 % | 0.848 | 0.853 | 0.929 |

The adapter helps only at the shallowest exits (+4-5.5 points at E = 1-2, where one attention
block adds context that a 1-2-layer state lacks) and costs about one extra layer per guess, so the
net effect on predicted speed is within ±0.03. Training loss kept falling while held-out accuracy
flattened after 4-5 epochs (~94k training positions), so more data would help a little. Not
repeated on 1.7B/4B: at +5 points of acceptance the 4B prediction would move from 1.12x to about
1.15x, still under the gate. The adapter path is verified lossless (0.6B, E = 1, kmax = 4, both
draft modes: 24 prompts, 0 real mismatches, 1.43 tokens/sweep).

### 4b. Top-2 alternatives in a spare row (`bench/tree_top2.py`)

| model | E | top-1 | top-2 | best predicted speedup | share of cycles using the 2-branch tree |
|---|---|---|---|---|---|
| 1.7B | 2 | 31.3% | 40.4% | 1.085 | 0% |
| 1.7B | 1 | 27.9% | 37.1% | 1.064 | 0% |
| 0.6B | 1 | 29.1% | 38.6% | 0.977 | 0% |

The full model's token is in the exit head's top-2 for 37-40 % of positions, against 28-31 % for
top-1 (+9 points). The best setting never uses the tree (0 % of cycles): on MLX a third row
costs 0.25-0.45 of a sweep more than a second (c(3) vs c(2)), and nine points of acceptance do
not pay for it. On llama.cpp, where rows 2-8 are nearly flat, a tree is cheaper per extra row, but
there the first extra row already costs 0.4-0.5 of a sweep. (This replay treats every held-out
position as a cycle start, so its tokens/sweep is slightly higher than the sequence replay of step 2.)

### 4c / 4d

Not run. 4c (draft-only LoRA on layers 1..E) makes the verify sweep recompute layers 1..E for
every drafted row, which removes the reuse of the draft's layer-1..E states that the layer-hot
schedule depends on; it only pays if acceptance rises well above what 4a reached. 4d
(LayerSkip-style tuning of the model itself) changes outputs and needs full-model training with
an early-exit loss behind a perplexity gate; full fine-tuning of 1.7B/4B does not fit this 16 GB
machine, and a LoRA version is a separate project. Both are only worth starting if the hybrid with
a DFlash drafter (step 6) shows the exit head adds tokens on top of DFlash.

## Step 5: head-to-head on this M4 (`bench/head_to_head.py`)

`results/20261114-234015-head_to_head` (analysis: `bench/h2h_analysis.py`). Qwen3-4B, T = 0,
256 new tokens, 6 held-out prompts (3 code, 3 chat explanations), 5 rounds, every arm a separate
process per round (load, warm-up, one generation per prompt), arms rotated, 60 s cool-down.
Rounds 0-1 ran late at night, rounds 2-4 the next morning (the run was paused in between; the
interrupted round was discarded and rerun). No round tripped a discard rule.

Arms: `mlx_serial` (mlx-lm greedy, MLX 4-bit); `ours` (layer-hot loop, affine exit at E = 1,
kmax = 2, τ = 0.3, MLX 4-bit); `dflash_mlx` (dflash-mlx 0.1.8 + `z-lab/Qwen3-4B-DFlash-b16`,
MLX 4-bit target); `llama_serial` (llama.cpp Q4_0, Metal, few-row kernels); `llama_dflash`
(+ the same DFlash drafter as bf16 GGUF, `--spec-draft-n-max 15`); `llama_dflash2`
(+ `mgoin/Qwen3-4B-speculator.dflash2` as bf16 GGUF, `--spec-draft-n-max 7`).

| arm | tok/s code | tok/s prose | tok/s all | vs own serial, same round: code | prose | all | tokens per target pass: code | prose | tok/s rounds 0-1 | tok/s rounds 2-4 |
|---|---|---|---|---|---|---|---|---|---|---|
| mlx_serial | 39.2 | 37.4 | 38.3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 40.5 | 33.5 |
| ours | 36.7 | 33.0 | 34.9 | 0.93 | 0.89 | **0.91** | 1.23 | 1.15 | 37.9 | 30.2 |
| dflash_mlx | 42.2 | 28.3 | 35.3 | 1.01 | 0.72 | 0.86 | 2.97 | 2.14 | 35.9 | 28.8 |
| llama_serial | 38.1 | 36.2 | 37.2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 35.4 | 38.2 |
| llama_dflash | 44.0 | 22.2 | 33.1 | 1.15 | 0.60 | 0.87 | 4.46 | 2.51 | 22.9 | 29.8 |
| llama_dflash2 | 40.3 | 33.7 | 37.0 | 1.15 | 1.02 | **1.09** | 3.23 | 2.84 | 35.7 | 41.2 |

"vs own serial, same round" divides each run by the same backend's serial run of the same prompt
in the same round, which cancels drift between rounds. "Tokens per target pass" is independent of
clock speed (ours: tokens per sweep; dflash-mlx: 1 / (1 - share of tokens from the draft);
llama.cpp: generated / (generated - accepted draft tokens)).

**Caveats.**
- Throttling. This fanless Air slows its GPU by 15-25 % after a few minutes of load, and nothing
  readable without sudo reports it (`pmset -g therm` stays clean), so the discard rules cannot
  catch it. The arms run in a fixed cyclic order, so an arm that follows its serial baseline runs
  a little hotter; the MLX arms ran early in the night rounds and late in the morning rounds, which
  is the night/morning gap in the last two columns. The A-B-B-A check of step 3 removes the
  position effect for `ours`: 0.86x, lower than the 0.91x here.
- Exactness. Against their own serial arm, `ours` diverges on 4/6 prompts and `dflash_mlx` on
  5/6 (both MLX batched-verify near-ties, see step 3); each llama.cpp speculative arm on 1/6
  (prompt 7 at token 108, not inspected). The plan asked that DFlash arms match serial text before
  their speed counts; with MLX's tie-breaking that would leave 1-2 prompts per MLX arm, so all
  prompts are counted and the divergences are reported instead.
- DFlash2 arm. The only public DFlash2 drafter for Qwen3-4B is a vLLM "speculators" export
  (nested `transformer_layer_config`, `sample_from_anchor: false`). llama.cpp's converter does not
  read that layout for DFlash2 (it does for DSpark), so the config was flattened the same way
  (`tools/dflash2_flat_config.json`); and llama.cpp's DFlash2 driver implements only the
  sample-from-anchor block layout. It runs, stays exact, and reaches 2.8-3.2 tokens per target pass,
  below the ~4.8 the drafter's own validation implies (3.8 accepted per block + 1). The cause
  (layout, Q4_0 target, prompt domain) was not isolated.

**Reading.**
- On this machine and model, nothing in the reference stack clearly beats plain decoding. The best
  arm is llama.cpp DFlash2 at 1.09x of llama.cpp serial (1.15x on code, 1.02x on prose); DFlash
  helps on code (1.0-1.15x) and hurts on prose (0.6-0.72x). The plan's reference numbers (2.0x
  prose, 3.4x code) are for a 27B target on an M3 Ultra.
- The DFlash arms accept 2-4.5 tokens per target pass, yet barely break even: their 1.07 GB bf16
  drafter (DFlash2: 2.8 GB including its own head) is read on every draft step, against 2.3 GB for
  the whole 4-bit target, on a bandwidth-bound laptop GPU; and verifying 8 rows costs 1.8x (llama.cpp)
  to 2.8x (MLX) a one-row step at ctx 512 (step 1); DFlash's 16-row blocks cost more.
- The layer-hot loop accepts 1.15-1.23 tokens per pass for almost no draft cost, which is far too
  few to pay for its two host round trips per cycle: 0.86-0.91x.

## Conclusions and next steps

1. **The hardware half of the idea holds on MLX.** A second token row through a hot weight sweep
   costs 2-11 % extra (c(2) = 1.02-1.11 on 0.6B-4B). On llama.cpp Metal the second row costs
   38-57 % and rows 2-8 are nearly flat; on the CPU path a second row costs ~50 %.
2. **The model half is the limit.** With the model frozen, the state after 1-8 layers predicts
   the full model's next token 27-40 % of the time (affine exit), almost all of it punctuation and
   function words. An attention adapter adds 4-5 points at E = 1-2; a top-2 alternative adds 9
   points but needs a third row. One guess per sweep at that rate caps out near 1.2-1.25 tokens per
   sweep, i.e. at most ~1.18x even with zero draft overhead.
3. **Gate.** Step 2 predicted 0.97x / 1.06x / 1.13x (0.6B / 1.7B / 4B), below the 1.2x gate on
   every model. The loop was built anyway (steps 4 and 5 need it) and is lossless up to bf16
   tie-breaking; measured on 4B it runs at 0.86x (A-B-B-A) to 0.91x (interleaved) of mlx-lm,
   because mlx-lm pipelines its steps and the loop needs two host round trips per cycle.
4. **The reference stack is not a strong baseline on this M4 + Qwen3-4B 4-bit.** DFlash/DFlash2
   land at 0.86-1.09x of serial despite 2-4.5 tokens per target pass, because a bf16 drafter is a
   large fraction of the bytes a 4-bit 4B target reads per step.

**Next steps (step 6), in order of expected value:**
1. Quantize the DFlash drafter (Q8_0 / Q4_0 GGUF, 4-bit MLX). On this bandwidth-bound machine the
   drafter's bytes are the main reason 3-4.5 accepted tokens per pass do not turn into speed; this
   is the cheapest test with the largest likely effect, and it improves the baseline the idea has
   to beat.
2. Exit head inside the DFlash verify batch (the hybrid of the plan). DFlash already verifies
   8-16 rows, where llama.cpp's c(K) is flat, so extra rows are nearly free there. The exit head
   costs one layer + a 16k head and can propose the alternative for a low-confidence block
   position (the step 4b tree, where it does pay: the extra row is free). Worth a replay first on
   DFlash's own acceptance traces (which position fails, and is the exit head's guess right there).
3. Only then the stand-alone layer-hot loop: a pipelined version (accept logic and KV bookkeeping
   on device, e.g. mask rejected rows instead of trimming; `mx.compile` for the draft path) or a
   llama.cpp port. Its ceiling is the ~1.12-1.18x predicted above, so it is not the route to beat
   the reference stack on its own.
4. 4c / 4d (draft-only LoRA, LayerSkip tuning) only if item 2 shows exit-head tokens adding to
   DFlash's.
