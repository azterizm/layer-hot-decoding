# Results

Raw evidence for every number in [../RESULTS.md](../RESULTS.md). One folder per run, named
`<date>-<time>-<what>`. Times are local (UTC+5).

## Files in a run folder

| File | What |
|---|---|
| `env.json` | Machine state at the start: hardware, power, thermal limit, idle CPU, memory, swap, disk, software versions, model files. Written by `bench/snapshot.py`. |
| `round_*.json`, `*_env.json` | Short machine snapshot before and after one round, with any discard reasons. |
| `summary.json` | The numbers the write-up quotes. |
| `raw.json` | Every timing before medians. |
| `acceptance.json`, `tree_top2.json`, `deep_head.json`, `hybrid_replay.json` | Offline results (accuracy, replay, predicted speed). |
| `*.log` | llama.cpp server logs, kept as printed. |
| `test.log` | Output of `tests/test_lossless.py` for that configuration. |

## Index by step

| Step | Folder | Script |
|---|---|---|
| 1. c(K), MLX | `20261114-192334-row_cost_mlx` (0.6B, 1.7B), `20261114-221454-row_cost_mlx_4B` | `bench/row_cost.py` |
| 1. c(K), llama.cpp | `20261114-225827-row_cost_llama` | `bench/row_cost_llama.py` |
| 2. a(E), exit head accuracy (and 4a adapter on 0.6B) | `20261114-215117-acceptance_0.6B`, `20261114-200238-acceptance_1.7B`, `20261114-225257-acceptance_4B` | `bench/acceptance.py` |
| 3. Lossless check | `20261115-083219-lossless_4B_affine_E1_k2_step`, `20261115-083553-lossless_1.7B_affine_E2_k2_step`, `20261115-083736-lossless_0.6B_adapter_E1_k4_chain` | `bench/lossless_run.py` |
| 3. Speed, A-B-B-A | `20261115-082716-ab_ours_4B` | `bench/ab_ours.py` |
| 4b. Top-2 in a spare row | `20261114-220605-tree_top2_1.7B_E2`, `20261114-220624-tree_top2_1.7B_E1`, `20261114-220643-tree_top2_0.6B_E1` | `bench/tree_top2.py` |
| 5. Head-to-head | `20261114-234015-head_to_head` (`summary.json`, `analysis.json`) | `bench/head_to_head.py`, `bench/h2h_analysis.py` |
| 6.1 Drafter precision | `20261115-093214-draft_quant` | `bench/draft_quant.py` |
| 6.1 Verify width | `20261115-102252-draft_quant` | `bench/draft_quant.py` |
| 6.2 Exit head inside DFlash (replay) | `20261115-091920-hybrid_replay_E1` (with DFlash traces), `20261115-091926-hybrid_replay_E4` | `bench/hybrid_replay.py` |
| 6.3 Pipelined loop, lossless | `20261115-092312-lossless_4B_affine_E1_k2_pipelined` | `bench/lossless_run.py` |
| 6.3 Pipelined loop, speed | `20261115-104800-ab_ours_4B_pipelined` (definitive), `20261115-092726-ab_ours_4B_pipelined` (short) | `bench/ab_ours.py` |
| 6.3 Where the time goes | `20261115-103414-profile_pipelined_4B`, `20261115-104056-cycle_parts_4B`, `20261115-110200-cycle_parts_4B` | `bench/profile_pipelined.py`, `bench/cycle_parts.py` |
| Acceptance route, Stage 0 (EAGLE-3, DFlash2 re-measured) | `20261116-144913-draft_quant` | `bench/draft_quant.py`, `tools/eagle3_baseline.sh` |
| Acceptance route, Stage 1a (deep heads) | `20261116-152329-deep_head_shallow1+deep+mid`, `..._frac0.25`, `..._frac0.5`, `20261116-153437-deep_head_deep`, `20261116-153921-deep_head_emb+deep+mid` | `layerhot/deep_head.py` |

## Superseded runs

Kept for the record, not quoted.

| Folder | Why |
|---|---|
| `20261114-185519-row_cost_mlx_0.6B` | First run, 0.6B only. Replaced by `20261114-192334-row_cost_mlx`. |
| `20261114-190147-acceptance_0.6B` | Same evaluation without the 4a adapter. Replaced by `20261114-215117-acceptance_0.6B`. |
| `20261114-190432-row_cost_llama` | 0.6B only. Replaced by `20261114-225827-row_cost_llama`. |
| `20261114-234015-head_to_head/incomplete_round2` | Round 2 was cut off by a pause and rerun. |

## Not in the repo

Hidden states, generated text, trained heads and model files live in `data/` and `~/models/gguf`
and are not committed (size). See [../docs/REPRODUCE.md](../docs/REPRODUCE.md) to regenerate them.
