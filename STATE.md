# Session state (2026-11-16): plan steps 1-6 done; acceptance route stopped at Gate 1

Plan: `~/.claude/plans/plan-to-execute-the-composed-cloud.md`. Full write-up: `RESULTS.md`.
Result line under the idea in
`~/Documents/personal-notes/Business/Technical Consultancy/Content Creation/SGLang Interventions/Core.md`.
Branch `poc/layer-hot` (main has no commits).

## Result directories (all with env.json)

| What | Dir |
|---|---|
| Step 1 MLX c(K) | `results/20261114-192334-row_cost_mlx` (0.6B, 1.7B), `results/20261114-221454-row_cost_mlx_4B` |
| Step 1 llama.cpp c(K) | `results/20261114-225827-row_cost_llama` (earlier 190432/185519 runs superseded) |
| Step 2 a(E) | `results/20261114-215117-acceptance_0.6B` (incl. 4a adapters), `...200238-acceptance_1.7B`, `...225257-acceptance_4B` |
| Step 3 lossless | `results/20261115-0832*-lossless_*`, `...092312-lossless_4B_affine_E1_k2_pipelined` |
| Step 3 speed (A-B-B-A) | `results/20261115-082716-ab_ours_4B` |
| Step 4b | `results/*tree_top2*` |
| Step 5 | `results/20261114-234015-head_to_head` (`summary.json`, `analysis.json`) |
| Step 6.1 drafter precision / verify width | `results/20261115-093214-draft_quant`, `results/20261115-102252-draft_quant` |
| Step 6.2 hybrid replay | `results/20261115-091920-hybrid_replay_E1` (with DFlash traces), `...091926-hybrid_replay_E4` |
| Acceptance route Stage 0 (EAGLE-3, DFlash2 Q8 re-measured) | `results/20261116-144913-draft_quant` |
| Acceptance route Stage 1a (deep heads) | `results/20261116-15*-deep_head_*` |
| Step 6.3 pipelined loop | `results/20261115-104800-ab_ours_4B_pipelined` (definitive), `...092726-ab_ours_4B_pipelined`, `...103414-profile_pipelined_4B`, `...104056-cycle_parts_4B`, `...110200-cycle_parts_4B` |

## Open question

Pipelined loop: GPU work per guess cycle is 1.07-1.09 one-row steps (isolated, hot or cool), but in
the chained loop under sustained load a cycle takes ~1.36 steps (cool: 1.03-1.09). Likely host-side
(graph build + Metal encoding on throttled CPU cores). Needs `powermetrics`/Instruments (sudo) or
`mx.compile` of the cycle to test.

## Not done (by decision, see RESULTS.md)

- 4c / 4d: exit head adds only +3-9 % tokens inside DFlash and 27-40 % top-1 alone.

## Disk

- Deleted 2026-11-16: all stored hidden states (`data/*/h_*.npy`), the bf16 DFlash/DFlash2 drafter
  GGUFs, the converter venv. Regenerate states with
  `python -m layerhot.collect --model <m> --exits 1 2 18`; reconverting drafters needs the raw HF
  checkpoints and a converter venv (see git history of this file for the exact commands).
- Kept in `~/models/gguf`: Q4_0 targets (0.6B/1.7B/4B), DFlash and DFlash2 drafters at Q8_0/Q4_0.
- EAGLE-3 baseline files deleted 2026-11-16. Rebuild with `tools/eagle3_baseline.sh --prepare`
  (pinned download, sha256-checked, recreates `.venv-convert`, converts and quantizes; tested: output
  GGUFs byte-identical to the measured ones), or without `--prepare` to also rerun the timed sweep. `data/*/` keeps generated text, labels, traces and trained heads.

## Environment notes

- Venvs: `.venv` (mlx 0.32.4, mlx-lm 0.32.2), `.venv-dflash` (dflash-mlx 0.1.8, + pyarrow).
  llama.cpp at `~/Code/llama.cpp` (`e3483e4`, built: bench, batched-bench, cli, server, quantize).
- llama-server here ignores per-request speculative settings (compiled out); compare spec vs serial
  with two servers side by side (`bench/draft_quant.py`).
- `mediaanalysisd` runs on AC power; benches wait for CPU idle >= 92 %.
- The fanless Air throttles 15-25 % under minutes of load with no sudo-free signal; compare A-B-B-A
  in one process, and use long runs: short cool runs overstate the pipelined loop by 15-30 %.
- Network ~0.7-1 MB/s; `tools/dl.sh` retries downloads; resume big files with `curl -C -`.
