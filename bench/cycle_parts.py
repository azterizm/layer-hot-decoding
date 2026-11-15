"""Step 6.3 diagnosis: time the parts of a pipelined layer-hot cycle in isolation (Qwen3-4B).

Through the same DeviceKVCache machinery, at a fixed ~400-token context, each part evaluated
synchronously N times after warm-up (median ms):
  one_row      embed + 36 layers + head, 1 row          (a plain decode step)
  two_rows     embed + 36 layers + head, 2 rows         (the verify alone, c(2))
  exit_head    affine exit head on one row
  draft_chain  layer 0 on row 0 -> exit head -> embed -> layer 0 on row 1 (the dependent chain)
  cycle        the full guess cycle (draft chain + layers 1..35 on 2 rows + head + accept logic)

Usage: python -m bench.cycle_parts [--ctx 400] [--iters 40]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

from bench import snapshot
from layerhot.collect import data_dir
from layerhot.exit_head import AffineExit
from layerhot.generate import greedy
from layerhot.model_utils import MODELS, load
from layerhot.pipelined import DeviceKVCache, PipelinedLayerHotDecoder


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=400)
    ap.add_argument("--iters", type=int, default=40)
    a = ap.parse_args(argv)
    out = snapshot.new_results_dir("cycle_parts_4B")
    snapshot.wait_quiet()
    env = {"short_before": snapshot.short_snapshot(), "software": snapshot.software(),
           "models": [snapshot.model_info(MODELS["4B"])], "config": vars(a)}
    s = load("4B")
    head = AffineExit.load(s, data_dir("4B") / "heads" / "affine_E1_V16384.npz").half()
    dec = PipelinedLayerHotDecoder(s, head, 1)
    N, E = s.n_layers, 1
    caches = [DeviceKVCache(a.ctx + 16, dec.n_kv, dec.head_dim) for _ in range(N)]
    dec._set(caches, 0, N, mx.array(0, dtype=mx.int32), a.ctx)
    mx.eval(s.layers(s.embed(mx.random.randint(1000, 50000, (1, a.ctx))), caches, 0, N))
    base = mx.array(a.ctx, dtype=mx.int32)
    ub = a.ctx
    y = mx.array([1234], dtype=mx.int32)

    def one_row():
        dec._set(caches, 0, N, base, ub + 1)
        return greedy(s.head(s.norm(s.layers(s.embed(y[None]), caches, 0, N))))

    def two_rows():
        dec._set(caches, 0, N, base, ub + 2)
        return greedy(s.head(s.norm(s.layers(s.embed(mx.array([[1234, 5678]])), caches, 0, N))))

    h_fixed = mx.random.normal((1, s.hidden)).astype(mx.bfloat16)

    def exit_head():
        return head.guess(h_fixed)

    def draft_chain():
        dec._set(caches, 0, E, base, ub + 1)
        h0 = s.layers(s.embed(y[None]), caches, 0, E)
        g, _ = head.guess(h0[:, -1])
        g = mx.maximum(g, 0).astype(mx.int32)
        dec._set(caches, 0, E, base + 1, ub + 2)
        return s.layers(s.embed(g[None]), caches, 0, E)

    def cycle():
        return dec._cycle(caches, y, base, ub)

    res = {}
    for name, fn in (("one_row", one_row), ("two_rows", two_rows), ("exit_head", exit_head),
                     ("draft_chain", draft_chain), ("cycle", cycle)):
        for _ in range(8):
            mx.eval(fn())
        ts = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            mx.eval(fn())
            ts.append((time.perf_counter() - t0) * 1e3)
        res[name] = statistics.median(ts)
        print(f"{name:12s} {res[name]:.3f} ms", flush=True)
    res["two_rows_over_one_row"] = res["two_rows"] / res["one_row"]
    res["cycle_over_one_row"] = res["cycle"] / res["one_row"]
    res["cycle_minus_two_rows"] = res["cycle"] - res["two_rows"]
    env["short_after"] = snapshot.short_snapshot()
    env["discard"] = snapshot.discard_reasons(env["short_before"], env["short_after"])
    snapshot.write_env(out, env)
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: round(v, 3) for k, v in res.items()}, indent=2), "\ndiscard:", env["discard"])
    print("results:", out)


if __name__ == "__main__":
    main()
