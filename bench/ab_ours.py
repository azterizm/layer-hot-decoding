"""A-B-B-A check of the layer-hot loop against mlx-lm greedy, one process, one loaded model.

The head-to-head runs arms in a fixed cyclic order, so on this fanless laptop the arm that runs
later in a round runs on a hotter GPU. Here each prompt runs serial, ours, ours, serial back to
back after a warm-up that brings the GPU to a steady state; the ratio per prompt is
mean(ours) / mean(serial). Also reports the synchronous one-row step time, to separate the
loop's own cost from mlx-lm's pipelining.

Usage: python -m bench.ab_ours --model 4B --exit 1 --kmax 2 --tau 0.3 [--max-tokens 256]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

from bench import snapshot
from layerhot import prompts as P
from layerhot.collect import data_dir
from layerhot.exit_head import AffineExit
from layerhot.generate import LayerHotDecoder, mlx_greedy
from layerhot.model_utils import MODELS, load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="4B")
    ap.add_argument("--exit", type=int, default=1)
    ap.add_argument("--kmax", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--mode", default="step")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup-s", type=int, default=90)
    ap.add_argument("--n-code", type=int, default=3)
    ap.add_argument("--n-prose", type=int, default=3)
    ap.add_argument("--passes", type=int, default=1, help="extra passes run the prompts in reverse order")
    a = ap.parse_args(argv)
    out = snapshot.new_results_dir(f"ab_ours_{a.model}_{a.mode}")
    snapshot.wait_quiet()
    env = {"short_before": snapshot.short_snapshot(), "software": snapshot.software(),
           "models": [snapshot.model_info(MODELS[a.model])], "config": vars(a)}
    s = load(a.model)
    head = AffineExit.load(s, data_dir(a.model) / "heads" / f"affine_E{a.exit}_V16384.npz").half()
    if a.mode == "pipelined":
        from layerhot.pipelined import PipelinedLayerHotDecoder

        dec = PipelinedLayerHotDecoder(s, head, a.exit)
    else:
        dec = LayerHotDecoder(s, head, a.exit, a.kmax, a.tau, mode=a.mode)
    test = [it for it in P.build() if it["split"] == "test"]
    items = [it for it in test if it["kind"] == "code"][:a.n_code] + [it for it in test if it["kind"] == "chat"][:a.n_prose]
    ids = [P.encode(s.tokenizer, it) for it in items]

    # warm-up to a steady thermal state
    t0 = time.time()
    while time.time() - t0 < a.warmup_s:
        mlx_greedy(s, ids[0], 128)
        dec.generate(ids[0], 128)

    rows = []
    order = []
    for k in range(a.passes):
        pairs = list(zip(items, ids))
        order += pairs if k % 2 == 0 else pairs[::-1]
    for it, x in order:
        r = {"id": it["id"], "kind": "code" if it["kind"] == "code" else "prose"}
        ref, a1 = mlx_greedy(s, x, a.max_tokens)
        o1, st1 = dec.generate(x, a.max_tokens)
        o2, st2 = dec.generate(x, a.max_tokens)
        _, a2 = mlx_greedy(s, x, a.max_tokens)
        r.update({"serial_tok_s": [a1, a2], "ours_tok_s": [st1.tok_s, st2.tok_s],
                  "ratio": (st1.tok_s + st2.tok_s) / (a1 + a2),
                  "tokens_per_sweep": st1.tokens_per_sweep, "identical": o1 == ref})
        rows.append(r)
        print(r, flush=True)
        mx.clear_cache()

    # synchronous single-row step (what step 1 called a sweep) at ~prompt + 128 context
    cache = s.make_cache()
    mx.eval(s.forward(mx.array([ids[0] + [1000] * 128]), cache))
    ts = []
    for _ in range(40):
        t1 = time.perf_counter()
        mx.eval(s.forward(mx.array([[1000]]), cache))
        ts.append((time.perf_counter() - t1) * 1e3)
        s.trim(cache, 1)
    sync_ms = statistics.median(ts)

    env["short_after"] = snapshot.short_snapshot()
    env["discard"] = snapshot.discard_reasons(env["short_before"], env["short_after"])
    snapshot.write_env(out, env)
    summ = {"rows": rows, "sync_step_ms": sync_ms,
            "serial_ms_per_token": 1e3 / statistics.mean(v for r in rows for v in r["serial_tok_s"]),
            "ours_ms_per_token": 1e3 / statistics.mean(v for r in rows for v in r["ours_tok_s"])}
    for k in ("code", "prose"):
        summ[f"ratio_{k}"] = statistics.mean(r["ratio"] for r in rows if r["kind"] == k)
    summ["ratio_all"] = statistics.mean(r["ratio"] for r in rows)
    rs = [r["ratio"] for r in rows]
    summ.update({"ratio_min": min(rs), "ratio_max": max(rs), "ratio_stdev": statistics.stdev(rs) if len(rs) > 1 else 0.0,
                 "n_measurements": len(rs), "share_above_1": sum(x > 1 for x in rs) / len(rs)})
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=2), "\ndiscard:", env["discard"])
    print("results:", out)


if __name__ == "__main__":
    main()
