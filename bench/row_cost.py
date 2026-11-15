"""Step 1 (MLX): cost of K new rows through one decode sweep, c(K) = t(K) / t(1).

For each model and KV context length C, times:
  - full sweep (embed + all layers + norm + full LM head) with K new rows
  - body only (no LM head) with K rows
  - first E layers alone (the draft path) with 1 row
  - full LM head alone, and reduced-vocab heads (8k, 16k rows), with K rows

Each timed call appends K rows to a cache prefilled to C tokens and trims them
again, so the context stays at C. Rounds are interleaved over K and wrapped in
machine snapshots; a round that trips a discard rule is marked and excluded.

Usage: python -m bench.row_cost --models 0.6B 1.7B 4B [--rounds 3] [--cooldown 60]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

import mlx.core as mx

from bench import snapshot
from layerhot.model_utils import MODELS, load

KS = [1, 2, 3, 4, 8]
ES = [1, 2, 4]
CONTEXTS = [512, 2048]  # overridable with --contexts


def timed(fn, iters: int) -> list[float]:
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def prefill(s, n_ctx: int, seed: int = 0):
    rng = random.Random(seed)
    toks = mx.array([[rng.randrange(1000, 100000) for _ in range(n_ctx)]])
    cache = s.make_cache()
    for i in range(0, n_ctx, 512):
        h = s.layers(s.embed(toks[:, i:i + 512]), cache, 0, s.n_layers)
        mx.eval(h, [c.state for c in cache])
    return cache


def bench_model(name: str, rounds: int, iters: int, warmup: int, cooldown: int, out_dir):
    s = load(name)
    L = s.n_layers
    vocab = s.model.args.vocab_size
    reduced = {n: s.head_rows(mx.arange(n)) for n in (8192, 16384)}
    for h in reduced.values():
        mx.eval(h.parameters())
    rows = []
    rng = random.Random(1)

    for n_ctx in CONTEXTS:
        cache = prefill(s, n_ctx)

        def sweep(K, with_head=True):
            toks = mx.array([[rng.randrange(1000, 100000) for _ in range(K)]])

            def f():
                h = s.layers(s.embed(toks), cache, 0, L)
                y = s.head(s.norm(h)) if with_head else h
                mx.eval(y)
                s.trim(cache, K)
            return f

        def draft(E):
            tok = mx.array([[rng.randrange(1000, 100000)]])

            def f():
                h = s.layers(s.embed(tok), cache, 0, E)
                mx.eval(h)
                s.trim(cache, 1, 0, E)
            return f

        def head_only(K, which):
            x = mx.random.normal((1, K, s.hidden)).astype(mx.bfloat16)
            mx.eval(x)
            fn = s.head if which == "full" else reduced[which]

            def f():
                mx.eval(fn(s.norm(x)))
            return f

        cases = []
        for K in KS:
            cases.append(("sweep", K, sweep(K)))
            cases.append(("body", K, sweep(K, with_head=False)))
            cases.append(("head_full", K, head_only(K, "full")))
        for E in ES:
            cases.append((f"layers_E{E}", 1, draft(E)))
        for n in reduced:
            cases.append((f"head_{n // 1024}k", 1, head_only(1, n)))

        for r in range(rounds):
            if r:
                time.sleep(cooldown)
            snapshot.wait_quiet()
            before = snapshot.short_snapshot()
            order = cases[:]
            random.Random(r).shuffle(order)
            res = {}
            for kind, K, f in order:
                timed(f, warmup)
                res[(kind, K)] = timed(f, iters)
            after = snapshot.short_snapshot()
            reasons = snapshot.discard_reasons(before, after)
            for (kind, K), ts in res.items():
                rows.append({"model": name, "ctx": n_ctx, "round": r, "kind": kind, "K": K,
                             "median_ms": statistics.median(ts), "min_ms": min(ts),
                             "p90_ms": sorted(ts)[int(0.9 * len(ts))], "n": len(ts),
                             "discard": reasons})
            (out_dir / f"round_{name}_{n_ctx}_{r}.json").write_text(
                json.dumps({"before": before, "after": after, "discard": reasons}, indent=2))
            print(f"{name} ctx={n_ctx} round={r} discard={reasons}", flush=True)
        del cache

    info = {"layers": L, "vocab": vocab, "hidden": s.hidden,
            "weights": snapshot.model_info(MODELS[name])}
    del s
    mx.clear_cache()
    return rows, info


def summarize(rows: list[dict], include_discarded: bool) -> list[dict]:
    keep = [r for r in rows if include_discarded or not r["discard"]]
    groups: dict = {}
    for r in keep:
        groups.setdefault((r["model"], r["ctx"], r["kind"], r["K"]), []).append(r["median_ms"])
    med = {k: statistics.median(v) for k, v in groups.items()}
    table = []
    for (m, ctx, kind, K), t in sorted(med.items()):
        if kind != "sweep":
            continue
        t1 = med[(m, ctx, "sweep", 1)]
        table.append({
            "model": m, "ctx": ctx, "K": K, "sweep_ms": round(t, 3), "c(K)": round(t / t1, 3),
            "body_c(K)": round(med[(m, ctx, "body", K)] / med[(m, ctx, "body", 1)], 3),
            "head_share": round(med[(m, ctx, "head_full", K)] / t, 3),
        })
    extras = []
    for (m, ctx, kind, K), t in sorted(med.items()):
        if kind.startswith("layers_E") or kind.startswith("head_") and K == 1:
            t1 = med[(m, ctx, "sweep", 1)]
            extras.append({"model": m, "ctx": ctx, "what": kind, "ms": round(t, 3),
                           "share_of_sweep": round(t / t1, 3)})
    return table, extras


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["0.6B", "1.7B", "4B"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--idle", type=int, default=60)
    ap.add_argument("--tag", default="row_cost_mlx")
    ap.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    a = ap.parse_args(argv)
    CONTEXTS[:] = a.contexts

    out = snapshot.new_results_dir(a.tag)
    env = snapshot.full_snapshot(a.idle, [MODELS[m] for m in a.models])
    snapshot.write_env(out, env)
    print("results:", out, flush=True)

    all_rows, infos = [], {}
    for i, m in enumerate(a.models):
        if i:
            time.sleep(a.cooldown)
        rows, info = bench_model(m, a.rounds, a.iters, a.warmup, a.cooldown, out)
        all_rows += rows
        infos[m] = info
    table, extras = summarize(all_rows, include_discarded=False)
    discarded_all = not table
    if discarded_all:
        table, extras = summarize(all_rows, include_discarded=True)
    (out / "raw.json").write_text(json.dumps(all_rows, indent=2))
    (out / "summary.json").write_text(json.dumps(
        {"models": infos, "c_of_K": table, "parts": extras,
         "all_rounds_discarded": discarded_all}, indent=2))
    print(json.dumps({"all_rounds_discarded": discarded_all}, indent=2))
    for r in table:
        print(r)
    for r in extras:
        print(r)


if __name__ == "__main__":
    main()
