"""Step 6.3 profile: where does a pipelined layer-hot cycle spend its time?

One process, one loaded Qwen3-4B, GPU warmed to a steady state, then per prompt the order
A B C C B A with
  A = mlx-lm greedy (`generate_step`, pipelined)
  B = PipelinedLayerHotDecoder(guess=False): plain one-row decoding through the same device-side
      KV cache, masks and pipelining as the loop (isolates this machinery's cost)
  C = PipelinedLayerHotDecoder(guess=True): the layer-hot loop
B/A = cost of the device-side cache machinery; C per cycle / B per cycle = cost of the guess row
(+ split first layer + exit head).

Usage: python -m bench.profile_pipelined [--max-tokens 256] [--warmup-s 90]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from bench import snapshot
from layerhot.collect import data_dir
from layerhot.exit_head import AffineExit
from layerhot.generate import mlx_greedy
from layerhot.model_utils import MODELS, load
from layerhot.pipelined import PipelinedLayerHotDecoder

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup-s", type=int, default=90)
    ap.add_argument("--exit", type=int, default=1)
    a = ap.parse_args(argv)
    out = snapshot.new_results_dir("profile_pipelined_4B")
    snapshot.wait_quiet()
    env = {"short_before": snapshot.short_snapshot(), "software": snapshot.software(),
           "models": [snapshot.model_info(MODELS["4B"])], "config": vars(a)}
    s = load("4B")
    head = AffineExit.load(s, data_dir("4B") / "heads" / f"affine_E{a.exit}_V16384.npz").half()
    B = PipelinedLayerHotDecoder(s, head, a.exit, guess=False)
    C = PipelinedLayerHotDecoder(s, head, a.exit, guess=True)
    prompts = json.loads((REPO / "results" / "20261114-234015-head_to_head" / "prompts.json").read_text())

    def run_a(ids):
        toks, tps = mlx_greedy(s, ids, a.max_tokens)
        return {"ms_per_token": 1e3 / tps, "tokens": toks}

    def run_pl(dec, ids):
        toks, st = dec.generate(ids, a.max_tokens)
        return {"ms_per_token": 1e3 * st.decode_s / st.tokens, "ms_per_cycle": 1e3 * st.decode_s / st.cycles,
                "tokens_per_cycle": st.tokens_per_sweep, "tokens": toks}

    t0 = time.time()
    while time.time() - t0 < a.warmup_s:
        run_a(prompts[0]["ids"])
        run_pl(C, prompts[0]["ids"])

    rows = []
    for p in prompts:
        ids = p["ids"]
        a1, b1, c1 = run_a(ids), run_pl(B, ids), run_pl(C, ids)
        c2, b2, a2 = run_pl(C, ids), run_pl(B, ids), run_a(ids)
        r = {"id": p["id"], "kind": p["kind"],
             "A_ms_per_token": statistics.mean([a1["ms_per_token"], a2["ms_per_token"]]),
             "B_ms_per_token": statistics.mean([b1["ms_per_token"], b2["ms_per_token"]]),
             "C_ms_per_cycle": statistics.mean([c1["ms_per_cycle"], c2["ms_per_cycle"]]),
             "C_ms_per_token": statistics.mean([c1["ms_per_token"], c2["ms_per_token"]]),
             "C_tokens_per_cycle": c1["tokens_per_cycle"],
             "B_identical_to_A": b1["tokens"] == a1["tokens"], "C_identical_to_A": c1["tokens"] == a1["tokens"]}
        r["B_over_A"] = r["B_ms_per_token"] / r["A_ms_per_token"]
        r["C_cycle_over_B_step"] = r["C_ms_per_cycle"] / r["B_ms_per_token"]
        r["C_cycle_over_A_step"] = r["C_ms_per_cycle"] / r["A_ms_per_token"]
        r["speedup_C_vs_A"] = r["A_ms_per_token"] / r["C_ms_per_token"]
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        mx.clear_cache()

    env["short_after"] = snapshot.short_snapshot()
    env["discard"] = snapshot.discard_reasons(env["short_before"], env["short_after"])
    snapshot.write_env(out, env)
    summ = {k: statistics.mean(r[k] for r in rows) for k in
            ("A_ms_per_token", "B_ms_per_token", "C_ms_per_cycle", "C_tokens_per_cycle", "B_over_A",
             "C_cycle_over_B_step", "C_cycle_over_A_step", "speedup_C_vs_A")}
    for k in ("code", "prose"):
        summ[f"speedup_C_vs_A_{k}"] = statistics.mean(r["speedup_C_vs_A"] for r in rows if r["kind"] == k)
    (out / "summary.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=2))
    print(json.dumps(summ, indent=2), "\ndiscard:", env["discard"])
    print("results:", out)


if __name__ == "__main__":
    main()
