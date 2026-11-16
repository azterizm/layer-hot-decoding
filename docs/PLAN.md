# Layer-hot token parallelism: proof of concept plan

**Status (2026-11-16):** steps 1-6 done, then the follow-up in [PLAN_acceptance.md](PLAN_acceptance.md), which stopped at its Gate 1. Results in [../RESULTS.md](../RESULTS.md). This is the plan as written on 2026-11-14, before any measurement.

## Context

The idea: while a layer's weights are hot, compute that layer for token 1 and also for token 2, seeding token 2 from token 1's layer-1 output. Both tokens ride the same weight sweep, the final head runs once for both, a separate check discards token 2 if its seed was wrong, and at least one token always comes out per sweep.

Goal: beat the reference stack (ANE-resident DFlash + Mirror-SD async + Wide-M Metal GPU) with no noticeable accuracy loss, on small dense Qwen models, all local.

Decisions already made:
- **Baseline to beat:** the runnable part, measured on this M4 16GB: llama.cpp master with DFlash2 and the few-row ("Wide-M") Metal kernels (PR ggml-org/llama.cpp#29869), plus `dflash-mlx`. ANE and Mirror-SD async are later additions.
- **Accuracy rule:** frozen model first. Version 1 trains only a small exit head, and every guess is verified by the full model, so output is token-identical to plain greedy decoding. Layer tuning comes later, behind an accuracy gate.
- **Version 1 scope:** no MTP, no ANE.

## What the idea reduces to

Two separate claims, each testable on its own:

- **Claim A (hardware):** a second token row through a hot layer costs much less than a second sweep. Cost curve `c(K)` = time of a K-row sweep / time of a 1-row sweep.
- **Claim B (model):** the state after the first `E` layers of token t predicts token t+1 often enough. Acceptance `a(E)` = how often the shallow guess equals the full model's own next token.

Speedup ≈ (accepted tokens per sweep) / (`c(K)` + draft cost). Both numbers are measurable before any real engine work.

Known reference points (not yet measured here):

| Source | Number |
|---|---|
| Wide-M kernels, M3 Ultra, Qwen3.8-27B Q4_0 (PR #29869) | `c(2)`≈1.32, `c(4)`≈1.73, `c(8)`≈2.2 |
| DFlash2 + Wide-M, same PR, T=0 | 2.0x prose, 3.4x code |
| Kangaroo (frozen model, exit at layer 2-3, adapter) | up to 1.68x |
| LayerSkip (tuned layers) | up to 2.16x |

**Expectation, stated plainly:** an exit after layer 1 on a frozen model will likely give 1.0-1.3x, mostly from grammar tokens. Deeper exits with an adapter usually reach about 1.5x. That is below the DFlash2 baseline. The realistic route to beating the reference is step 6: use the exit head together with a DFlash drafter, not instead of it. Steps 1-3 tell us cheaply whether that route is worth taking.

## Execution substrate

- **MLX (`mlx-lm`) for steps 1-4.** Per-layer access is simple, wall-clock is real Apple GPU time, and head training is small enough for MLX. New repo, `uv` venv on Python 3.12 (system Python is 3.14, MLX not installed).
- **llama.cpp master for the baseline and the final port.** It has the Wide-M kernels and DFlash2. The M4 has no Metal tensor API, so the few-row kernels are the active path.
- PyTorch only as a fallback for training if MLX training is awkward.

Disk: 24 GB free. Keep to 4-bit weights; delete each model after its step if space gets tight.

## Machine state record (before every measurement)

Machine: MacBook Air M4 (Mac16,12), 16 GB, fanless. It throttles under sustained load, so every number is only comparable with a recorded state next to it.

`bench/snapshot.py` writes `results/<timestamp>/env.json`. It is written as part of step 1 and called by every bench script. Everything is read without `sudo`.

**Once per session (60 s idle sample, nothing else running):**

| What | How |
|---|---|
| Model, chip, RAM, macOS version | `system_profiler SPHardwareDataType`, `sw_vers` |
| Power source, charge %, Low Power Mode | `pmset -g batt`, `pmset -g` |
| Battery cycle count, capacity, condition | `system_profiler SPPowerDataType` |
| Idle CPU % (mean and max over 60 s), load average | `top -l 30 -s 2 -n 0`, `sysctl vm.loadavg` |
| Top 5 processes by CPU and by memory | `ps -Ao pcpu,pmem,rss,comm -r` |
| Memory free %, wired/compressed pages, swap used | `memory_pressure`, `vm_stat`, `sysctl vm.swapusage` |
| Idle GPU utilisation and GPU memory in use | `ioreg -r -d 1 -c IOAccelerator` (PerformanceStatistics), if exposed |
| Idle system power draw | `ioreg -rn AppleSmartBattery` (power telemetry), if exposed; otherwise left empty |
| Thermal limit | `pmset -g therm` (CPU speed limit) |
| Free disk, HF cache size, repo size | `df -h /System/Volumes/Data`, `du -sh` |
| Software | Python, `mlx`, `mlx-lm` versions; llama.cpp commit; model repo revision, quantisation and file hash |

**Before and after each benchmark round (short snapshot):** power source, thermal limit, memory free %, swap used, CPU idle %, timestamp.

**A round is discarded and rerun after a cool-down if any of these hold:**
- not on the charger, or Low Power Mode on
- CPU idle below 90% before the round starts
- thermal CPU speed limit below 100 before or after the round
- swap used grew during the round
- free disk below 5 GB

Run protocol: charger connected, lid open, browsers and other heavy apps closed, 60 s cool-down between rounds, arms interleaved (serial, ours, baseline, repeat).

Energy per token (a battery run) is not measured in this plan; it can be one extra round per arm after step 5.

## Steps

### 1. Measure `c(K)` on this M4 (no training, about an hour)
- First: write `bench/snapshot.py` and record the session baseline.
- MLX: Qwen3-0.6B, 1.7B, 4B (4-bit). Time one decode sweep with K = 1, 2, 3, 4, 8 new rows against a 512- and 2048-token KV cache. Also time `E` layers alone for E = 1, 2, 4, and the full LM head alone.
- llama.cpp: `llama-batched-bench -npl 1,2,3,4,8` on the same models (Metal), and once with `-ngl 0` to test the literal cache claim on CPU (one Q4 layer of 0.6B is about 7 MB; P-core L2 is 16 MB).
- Output: a table of `c(K)` per model and backend, and the share of a sweep spent in the LM head. On 0.6B the head is about a quarter of all weights, so the draft head must be a reduced-vocabulary one.
- **Gate:** if `c(2)` > 1.6 everywhere, a second row is too expensive and the idea needs the Wide-M path only; continue on llama.cpp numbers.

### 2. Measure `a(E)` offline (frozen model)
- Generate about 100k tokens of the target's own greedy output on a prompt mix (chat, prose, code). Acceptance is measured against the model's own choices, not against reference text. Wikitext is already in the HF cache for extra prose.
- Record hidden states after layers E ∈ {1, 2, 3, 4, 6, 8} and the final argmax.
- Exit heads, cheapest first:
  1. Logit lens (final norm + tied embedding rows), zero training.
  2. Trained affine map + reduced-vocab head (top 8-16k tokens seen in the target's output, plus an "abstain" class).
- Report: agreement with the full model per E; split by token class (function words, punctuation, sub-word continuations, content); precision vs coverage as the confidence threshold moves.
- **Gate:** combine with step 1 to predict speedup. ≥ 1.2x predicted → step 3 at that E. Below → go to step 4a/4b first, since the loop would only confirm a null.

### 3. The real loop, lossless greedy
`layerhot/generate.py`, one cycle:
1. Run layers 1..E on the last accepted token; exit head guesses the next token. If confident, embed the guess and repeat (up to K-1 guesses; stop at the first abstain).
2. Run layers E+1..N once over all K rows, reusing the layer 1..E states and KV entries from the draft.
3. One final norm + full LM head over K rows.
4. Accept the longest prefix where each guess equals the full model's token, plus one bonus token. Truncate the KV cache for rejected rows.

With E = 1 this is the schedule in the note. Always yields ≥ 1 token per sweep.

- Check: output token-identical to `mlx_lm` greedy on the prompt set. Any mismatch is inspected for near-tie logits (batched vs single-row float differences).
- Report: tok/s vs plain MLX, tokens per sweep, acceptance by token class.

### 4. Raise acceptance, cheapest first
- **4a.** Sweep E and add a Kangaroo-style adapter (one attention block on the layer-E state, distilled from the target's final distribution). Model frozen, still lossless.
- **4b.** Use spare rows for top-2 alternatives at low-confidence positions (tree verify).
- **4c.** Draft-only LoRA on layers 1..E: active while guessing, off while verifying. Verify recomputes those E layers, so output stays exact.
- **4d.** LayerSkip-style tuning of the model itself (the "layers learn to work alone" version). Changes outputs, so it ships only if wikitext perplexity moves ≤ 1% relative and a small task set stays within noise.

### 5. Head-to-head on this M4
- Model: the smallest dense Qwen that has a public DFlash/DFlash2 drafter and fits 16 GB (Qwen3-4B is the candidate; confirm on Hugging Face at that point).
- Arms: serial; `dflash-mlx`; llama.cpp DFlash2 + Wide-M (`--spec-type draft-dflash --spec-draft-n-max 7`); ours.
- Same prompts (code and prose), T=0, 64-256 generated tokens, median of 5 after warm-up, interleaved runs because the laptop GPU throttles.

### 6. Extend from what steps 1-5 show
Only sketched; the order depends on the measurements.
- Hybrid drafter: exit head takes the confident grammar tokens at near-zero cost, DFlash takes the rest.
- Port the loop to llama.cpp Metal to use the Wide-M kernels directly.
- Mirror-SD style: exit-head top-k from the verify sweep pre-seeds the drafter for the next block.
- DFlash on ANE with a drafter trained on the tuned model; MTP.

## Files

New repo:
- `bench/snapshot.py`, `bench/row_cost.py` (step 1)
- `results/<timestamp>/env.json` next to every result file
- `layerhot/collect.py`, `layerhot/exit_head.py`, `bench/acceptance.py` (step 2)
- `layerhot/generate.py`, `tests/test_lossless.py` (step 3)
- `bench/head_to_head.py` (step 5)
- `RESULTS.md`: one table per step, with the gate decision

Each file is created when its step starts.

## Verification

- Every reported number has an `env.json` beside it, and no round that tripped a discard rule is included.
- Step 1: `c(1)` = 1.0 by construction; llama.cpp serial tok/s matches `llama-bench` tg.
- Step 2: logit lens at the last layer must reproduce the model's argmax 100% (sanity check on the collection code).
- Step 3: `tests/test_lossless.py` passes (token-identical to greedy on all prompts), and tok/s is reported next to plain `mlx_lm.generate` on the same prompts.
- Step 5: DFlash arms produce the same T=0 text as serial before their speed counts.

## Risks

- Layer-1 acceptance on a frozen model may be too low to pay for the second row. Step 2 shows this before the loop is built.
- On 0.6B, framework overhead per call may dominate, hiding or inflating gains. Report 1.7B and 4B alongside.
- No DFlash2 drafter may exist for a model that fits 16 GB; then the head-to-head uses `dflash-mlx` with the nearest supported model.
