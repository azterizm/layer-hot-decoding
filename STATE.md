# Session state (2026-11-15 08:45 PKT): plan steps 1-5 complete

Plan: `~/.claude/plans/plan-to-execute-the-composed-cloud.md`. Full write-up: `RESULTS.md` (no placeholders left).
Result line added under the idea in
`~/Documents/personal-notes/Business/Technical Consultancy/Content Creation/SGLang Interventions/Core.md`.

## Result directories (all with env.json)

| What | Dir |
|---|---|
| Step 1 MLX c(K) | `results/20261114-192334-row_cost_mlx` (0.6B, 1.7B), `results/20261114-221454-row_cost_mlx_4B` |
| Step 1 llama.cpp c(K) | `results/20261114-225827-row_cost_llama` (earlier 190432/185519 runs superseded) |
| Step 2 a(E) | `results/20261114-215117-acceptance_0.6B` (incl. 4a adapters), `...200238-acceptance_1.7B`, `...225257-acceptance_4B` |
| Step 3 lossless | `results/20261115-0832*-lossless_*` (4B, 1.7B, 0.6B adapter) |
| Step 3 speed (A-B-B-A) | `results/20261115-082716-ab_ours_4B` |
| Step 4b | `results/*tree_top2*` |
| Step 5 | `results/20261114-234015-head_to_head` (`summary.json`, `analysis.json`) |

## Not done (by decision, see RESULTS.md)

- 4c / 4d, step 6. Suggested order: quantized DFlash drafter → exit head inside the DFlash verify
  batch → pipelined layer-hot loop / llama.cpp port.

## Disk cleanup candidates (not deleted; ask the user)

- `~/.cache/huggingface/hub/models--mgoin--Qwen3-4B-speculator.dflash2` (2.8 GB, already converted to GGUF)
- `data/4B/h_{1,2,4,8}.npy` (2.4 GB; heads trained, needed only to re-run 4B acceptance)
- `.venv-convert` (0.5 GB, torch for the GGUF converter)

## Environment notes

- Venvs: `.venv` (mlx 0.32.4, mlx-lm 0.32.2), `.venv-dflash` (dflash-mlx 0.1.8), `.venv-convert`.
  llama.cpp at `~/Code/llama.cpp` (`e3483e4`), GGUFs in `~/models/gguf`.
- `mediaanalysisd` runs on AC power; benches wait for CPU idle >= 92 %.
- The fanless Air throttles its GPU 15-25 % after minutes of load without any sudo-free signal;
  compare arms A-B-B-A in one process when the difference matters.
- Network ~0.7-1 MB/s; `tools/dl.sh` retries downloads; resume big files with `curl -C -`.
