# Acceptance route: plan

This route stopped at Gate 1 on 2026-11-16. Stage 0 found EAGLE-3 off the shelf at 0.64-0.82x, so
the bar stays at 1.22x. In Stage 1a the best head was right 29.4 % of the time on the first guess. It
was limited by training data, and the layer-hot input added nothing.
See [../RESULTS.md](../RESULTS.md), section "Acceptance route".

## Why this route

Steps 1-6 (see `RESULTS.md`) showed the hardware half of the idea holds on MLX (a second row costs
2-11 %), but the guess is right only 27-40 % of the time, which caps the layer-hot loop near
1.1x even with perfect engineering. The tuned reference stack already reaches 1.22x (llama.cpp
DFlash2, Q8_0 drafter; 1.52x on code with DFlash at 7 draft tokens). The only lever that can move
the layer-hot loop past that is acceptance.

Every exit head so far read only the *current* token's shallow state h_E(y).
But each sweep already finishes the *previous* token through all 36 layers, so its deep state
f(p-1) is free. EAGLE-style drafters use exactly that state plus the current token's embedding,
and reach ~60-80 % on the first guess in published results. A head that reads f(p-1) *and*
h_E(y) is EAGLE with one extra ingredient: the layer-hot one (the current token's state after E
layers instead of its raw embedding). It keeps the model frozen, so output stays lossless.

What acceptance is needed (4B, MLX, one guess per sweep, measured costs):

| first-guess acceptance | tokens / cycle | speed at today's sustained cycle cost (1.36 steps) | at GPU-only cost (1.09 steps) |
|---|---|---|---|
| 30 % (today) | 1.30 | 0.96x | 1.19x |
| 60 % | 1.60 | 1.18x | 1.47x |
| 75 % | 1.75 | **1.29x** | 1.61x |
| 75 % + 2nd chained guess at ~60 % | ~2.2 | ~1.4x (if chain cost stays ~1.6 steps) | ~1.7x |

So the bet pays only if a frozen-model head reaches roughly 70 %+ on the first guess, or the
host-side overhead is fixed, or both.

## Bars to beat

- llama.cpp DFlash2 with a Q8_0 drafter runs at 1.22x of its serial (1.31x code, 1.13x prose).
- llama.cpp DFlash with 7 draft tokens runs at 1.19x (1.52x on code).
- llama.cpp EAGLE-3 with the public `AngelSlim/Qwen3-4B_eagle3` head is not measured yet (Stage 0).

## Stages (each ends in a gate; stop at the first failed gate)

### Stage 0: baselines and data (no training, ~3-4 h, mostly downloads and timed runs)

- 0a. EAGLE-3 off the shelf: download `AngelSlim/Qwen3-4B_eagle3` (437 MB), recreate the converter
  venv, convert to GGUF (bf16 and Q8_0), run `bench/draft_quant.py`-style A-B-B-A against llama.cpp
  serial with `--spec-type draft-eagle3` at 3/5/7 draft tokens. Record tokens per pass and speed.
- 0b. Re-collect 4B features for training heads: for the ~120k positions already generated,
  store h_1, h_2 (shallow, current token), one mid layer, and the last layer (deep, previous token),
  ~2.5 GB. Generate more of the model's own output (target ~0.5-1M tokens with a larger prompt
  set) only if Stage 1 shows the head is data-limited; on this machine 4B generates ~100 tok/s
  batched, so 1M tokens is ~3 h of GPU.
- At Gate 0, if EAGLE-3 on llama.cpp is already >= 1.5x overall, the layer-hot head must beat that
  number, not 1.22x; re-state the targets before Stage 1.

### Stage 1: deep-context exit head, offline (frozen model, lossless; ~1 day)

Head inputs for guessing the token after y (position p): f(p-1) (last layer, previous token, from
the previous sweep), h_E(y) (this sweep, after E layers), optionally one mid-layer state of p-1.
Variants, cheapest first:
- 1a. MLP on [h_E(y), f(p-1)] -> frozen reduced-vocab head (+ abstain). Minutes to train.
- 1b. EAGLE-style one-layer decoder with its own KV over past positions, input fc([h_E(y),
  f(p-1), mid(p-1)]), trained with cross-entropy + KL to the full model, chained guesses fed
  by its own output state (as EAGLE does).
- 1c. Same as 1b with e(y) (embedding) instead of h_E(y): plain EAGLE inputs. The 1b-vs-1c
  difference is the measured value of the layer-hot ingredient itself.
- Report first-guess and chained acceptance by token class, replay tokens/cycle, and predicted
  speed with both the sustained (1.36-step) and GPU-only (1.09-step) cycle costs.
- To pass Gate 1, the predicted speed must be >= 1.3x at the sustained cycle cost *and* above the Stage 0 EAGLE-3 number.
  If acceptance stays below ~60 %, go to Stage 3 only if the gap looks closable by tuning;
  otherwise stop and adopt the reference stack.

### Stage 2: into the pipelined loop and measured (lossless; ~1 day)

- Head on device with its own KV cache, chained guesses (kmax 2-4) built inside the same graph
  (no host decisions), as in `layerhot/pipelined.py`.
- Spend the 1-2 h `mx.compile` timebox on the cycle here, since the draft head adds host work.
- `tests/test_lossless.py` (tie-window rule unchanged), then the definitive A-B-B-A
  (12 prompts x 2 passes, sustained load) against mlx-lm, and absolute tok/s next to llama.cpp
  DFlash2 Q8_0 and EAGLE-3 on the same prompts.
- Gate 2 means success, at >= 1.22x of mlx-lm under sustained load, which beats the tuned reference stack.

### Stage 3: only if acceptance is the limit (changes or extends the model; days)

- 3a. Draft-only LoRA on layer 0 (q, o, MLP; k/v frozen so the KV cache stays shared; the verify
  recomputes layer 0 for drafted rows). Still lossless.
- 3b. LayerSkip-style LoRA with an early-exit loss. Changes outputs: ships only if wikitext
  perplexity moves <= 1 % relative and a small task set stays within noise. Needs your explicit
  go-ahead at that point (full fine-tuning of 4B does not fit in 16 GB; LoRA on the 4-bit base does).

## Measurement rules (unchanged, plus lessons from step 6)

- Charger, quiet machine, `env.json` next to every number, discard rules as before.
- Speed claims only from long A-B-B-A runs after warm-up (short cool runs overstated the pipelined
  loop by 15-30 %); report the per-prompt range and how many measurements beat 1.0.

## Resources

- The disk has 12 GB free. This needs ~2.5 GB of features, 0.4 GB for the EAGLE-3 head and GGUFs,
  and 0.5 GB for the converter venv. If space runs short, delete the bf16 drafter GGUFs (3.9 GB; the
  Q8_0/Q4_0 copies are better).
- Stage 0 needs ~2 h of timed GPU runs and Stage 1 ~2-4 h of training (+3 h if more data is needed).
- The network runs at ~1 MB/s. The only large download is the EAGLE-3 head (~8 min).

## Risks

- EAGLE-3 may already beat everything; then the honest result is "use EAGLE-3", and Stage 1c-vs-1b
  tells whether the layer-hot ingredient adds anything on top.
- ~120k tokens may be too little for a one-layer decoder head; more data costs hours of generation.
- The sustained-load host overhead may grow with a heavier draft head; Stage 2's `mx.compile`
  timebox is the mitigation, a C++ path the fallback.
