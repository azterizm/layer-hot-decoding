# Reproduce

Everything ran on one MacBook Air M4 (Mac16,12), 16 GB, macOS 26.6.2. Apple silicon is required
(MLX). Numbers on another machine will differ; the method should carry over.

## Software

| What | Version |
|---|---|
| Python | 3.12.11 (via `uv`) |
| mlx / mlx-lm | 0.32.4 / 0.32.2 |
| dflash-mlx | 0.1.8 (separate venv) |
| llama.cpp | `e3483e40b02326383a54a23ae71f2bfa9f81181a` (2026-11-14 master) |

The full package lists are [../env/requirements-main.txt](../env/requirements-main.txt) and
[../env/requirements-dflash.txt](../env/requirements-dflash.txt).

## Setup

```sh
# main venv: MLX, mlx-lm, training, benches
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r env/requirements-main.txt

# dflash-mlx venv (step 5, 6.1, 6.2 only)
uv venv --python 3.12 .venv-dflash
uv pip install --python .venv-dflash/bin/python -r env/requirements-dflash.txt

# llama.cpp, built with Metal
git clone https://github.com/ggml-org/llama.cpp ~/Code/llama.cpp
git -C ~/Code/llama.cpp checkout e3483e40b02326383a54a23ae71f2bfa9f81181a
cmake -S ~/Code/llama.cpp -B ~/Code/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build ~/Code/llama.cpp/build -j --target llama-bench llama-batched-bench llama-cli llama-server llama-quantize
```

The scripts expect llama.cpp at `~/Code/llama.cpp` and GGUF files in `~/models/gguf`.

## Models

| Use | Source |
|---|---|
| MLX targets | `mlx-community/Qwen3-{0.6B,1.7B,4B}-4bit` (downloaded on first use) |
| llama.cpp targets | `unsloth/Qwen3-{0.6B,1.7B,4B}-GGUF`, Q4_0, saved as `~/models/gguf/Qwen3-<size>-Q4_0.gguf` |
| DFlash drafter | `z-lab/Qwen3-4B-DFlash-b16` |
| DFlash2 drafter | `mgoin/Qwen3-4B-speculator.dflash2`, converted with the flattened config in `tools/dflash2_flat_config.json` |
| EAGLE-3 head | `AngelSlim/Qwen3-4B_eagle3`, pinned in `tools/eagle3_baseline.sh` |
| Prose prompts | wikitext-103 from the Hugging Face cache |

`tools/dl.sh` retries slow downloads. `tools/eagle3_baseline.sh --prepare` downloads, converts and
quantizes the EAGLE-3 head with sha256 checks; the converted files were byte-identical to the
measured ones. Drafter GGUFs (DFlash, DFlash2) are converted with llama.cpp's
`convert_hf_to_gguf.py` and quantized with `llama-quantize` to Q8_0 / Q4_0.

## Commands by step

Run from the repo root with `.venv/bin/python` unless noted. Each command writes a new
`results/<timestamp>-<what>/` folder with `env.json`.

| Step | Command |
|---|---|
| Machine snapshot | `python -m bench.snapshot` |
| 1. c(K), MLX | `python -m bench.row_cost --models 0.6B 1.7B 4B` |
| 1. c(K), llama.cpp | `python -m bench.row_cost_llama --models 0.6B 1.7B 4B` |
| 2. Collect data | `python -m layerhot.collect --model 4B --exits 1 2 18` (default exits for 0.6B/1.7B) |
| 2. Exit heads + replay | `python -m bench.acceptance --model 4B --row-cost results/<ts>-row_cost_mlx_4B` (`--adapter` for 4a) |
| 3. Decode loop, one prompt | `python -m layerhot.generate --model 4B --exit 1 --kmax 2 --tau 0.3 --prompt "..."` |
| 3. Lossless check | `python -m bench.lossless_run --model 4B --exit 1 --kmax 2 --tau 0.3` |
| 3. Speed, A-B-B-A | `python -m bench.ab_ours --model 4B --exit 1 --kmax 2 --tau 0.3` |
| 4a. Adapter | `python -m layerhot.adapter --model 0.6B --exit 1 2 4 8` |
| 4b. Top-2 tree | `python -m bench.tree_top2 ...` (see the script header) |
| 5. Head-to-head | `python -m bench.head_to_head --arms mlx_serial ours dflash_mlx llama_serial llama_dflash llama_dflash2`, then `python -m bench.h2h_analysis results/<ts>-head_to_head` |
| 6.1 Drafters | `python -m bench.draft_quant --variants llama_dflash2_q8 llama_dflash_q8_n7 ...` |
| 6.2 Hybrid replay | `.venv-dflash/bin/python -m bench.hybrid_replay trace --n 40`, then `python -m bench.hybrid_replay replay --exit 1` |
| 6.3 Pipelined loop | `python -m bench.lossless_run --model 4B --exit 1 --kmax 2 --tau 0 --mode pipelined`, `python -m bench.ab_ours --mode pipelined`, `python -m bench.profile_pipelined`, `python -m bench.cycle_parts` |
| Stage 0, EAGLE-3 | `tools/eagle3_baseline.sh` (~35 min timed sweep) |
| Stage 1a, deep heads | `python -m layerhot.deep_head --inputs shallow1 deep mid` (also `deep`, `emb deep mid`, `--train-frac 0.25`) |
| Tables | `python -m bench.report --mlx ... --llama ... --acceptance ... --tree ...` |
| Unit check | `LH_MODEL=0.6B LH_EXIT=1 LH_KMAX=4 pytest` (env vars listed in `tests/test_lossless.py`) |

Each script's header has its full usage.

## Data not in the repo

`data/<model>/` holds generated text, labels, trained heads, DFlash traces and hidden states. It is
not committed. Hidden states (`h_*.npy`, ~2.5 GB for 4B) were deleted after use; regenerate them
with `python -m layerhot.collect --model <m> --exits 1 2 18`.

## Measurement rules

- On the charger, lid open, other heavy apps closed.
- `env.json` next to every number. A round is discarded if the machine was on battery, CPU idle was
  below 90 % before it, the thermal limit was below 100, swap grew, or free disk was below 5 GB.
  Benches wait for a quiet machine before each round.
- Speed claims come from long A-B-B-A runs in one process after a warm-up. Short runs on a cool GPU
  overstated the pipelined loop by 15-30 %.

## Quirks of the test machine

- The fanless Air slows its GPU 15-25 % after a few minutes of load. Nothing readable without sudo
  shows it (`pmset -g therm` stays clean). This is why speed is compared A-B-B-A.
- Photos' `mediaanalysisd` uses 2-3 cores for minutes at a time on AC power.
- The local llama-server build ignores per-request speculative settings, so serial and speculative
  are two servers side by side (`bench/draft_quant.py`).
- The network is slow, so set `HF_HUB_DISABLE_XET=1` and resume large files with `curl -C -`.
